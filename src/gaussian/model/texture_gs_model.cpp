#include "texture_gs_model.hpp"

#include <ranges>
#include <utility>

#include <ImGui/imgui.h>

#include "gs_model.hpp"
#include "ply.hpp"
#include "rasterizer/defines.hpp"
#include "rasterizer/texture_rasterizer.hpp"
#include "utils/camera/camera.hpp"
#include "utils/utils.hpp"

#include "rasterizer/rasterizer.hpp"

TextureGaussianModel::TextureGaussianModel(const char *geometryPlyPath, const char *appearancePlyPath,
                                           int sh_degree, int device)
    : GaussianModel(sh_degree, device), Model() {

  _loadPly(geometryPlyPath, appearancePlyPath);
  initMesh();
}

TextureGaussianModel::~TextureGaussianModel() {

  //
  cudaFree(_model_position_cuda);
  cudaFree(_model_normal_cuda);
  cudaFree(_model_texCoords_cuda);
  cudaFree(_model_tangent_cuda);
  cudaFree(_model_bitangent_cuda);
  cudaFree(_model_basecolor_map_cuda);
  cudaFree(_model_normal_map_cuda);
  cudaFree(_model_height_map_cuda);
  cudaFree(_model_roughness_map_cuda);
  cudaFree(_model_mask_filter_cuda);
  cudaFree(_appearance_face_idx_cuda);
  cudaFree(_selected_face_idx_cuda);

  //
  cudaFree(_view_cuda);
  cudaFree(_proj_cuda);
  cudaFree(_mask_cuda);
}

namespace {
template <typename T>
void cudaAllocCopy(float **cudaPointer, int count, const std::vector<T> &data, int countA,
                   const std::vector<T> &dataA) {

  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)cudaPointer, sizeof(T) * (count + countA)));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(*cudaPointer, data.data(), sizeof(T) * count, cudaMemcpyHostToDevice));

  auto sizeT = sizeof(T);
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(*cudaPointer + (sizeT / sizeof(float)) * count, dataA.data(),
                                   sizeof(T) * countA, cudaMemcpyHostToDevice));
}
} // namespace

void TextureGaussianModel::_loadPly(const char *geometryPlyPath, const char *appearancePlyPath) {

  // load ply
  // TODO: get degree from ply
  std::vector<Pos> pos;
  std::vector<Rot> rot;
  std::vector<Scale> scale;
  std::vector<float> opacity;
  std::vector<SHs<3>> shs;
  std::vector<Face> face;

  std::vector<Pos> posA;
  std::vector<Rot> rotA;
  std::vector<Scale> scaleA;
  std::vector<float> opacityA;
  std::vector<SHs<3>> shsA;
  std::vector<Face> faceA;
  std::vector<Coords> coords;
  std::vector<float> faceIds;
  glm::vec3 _sceneminA, _scenemaxA;
  if (_sh_degree == 0) {
    gsCount = loadGeometryPly<0>(geometryPlyPath, pos, shs, opacity, scale, rot, face, _scenemin, _scenemax);
    _gsCountA = loadAppearancePly<0>(appearancePlyPath, posA, shsA, opacityA, scaleA, rotA, coords, faceIds,
                                     _sceneminA, _scenemaxA);
  } else if (_sh_degree == 1) {
    gsCount = loadGeometryPly<1>(geometryPlyPath, pos, shs, opacity, scale, rot, face, _scenemin, _scenemax);
    _gsCountA = loadAppearancePly<1>(appearancePlyPath, posA, shsA, opacityA, scaleA, rotA, coords, faceIds,
                                     _sceneminA, _scenemaxA);
  } else if (_sh_degree == 2) {
    gsCount = loadGeometryPly<2>(geometryPlyPath, pos, shs, opacity, scale, rot, face, _scenemin, _scenemax);
    _gsCountA = loadAppearancePly<2>(appearancePlyPath, posA, shsA, opacityA, scaleA, rotA, coords, faceIds,
                                     _sceneminA, _scenemaxA);
  } else if (_sh_degree == 3) {
    gsCount = loadGeometryPly<3>(geometryPlyPath, pos, shs, opacity, scale, rot, face, _scenemin, _scenemax);
    _gsCountA = loadAppearancePly<3>(appearancePlyPath, posA, shsA, opacityA, scaleA, rotA, coords, faceIds,
                                     _sceneminA, _scenemaxA);
  } else {
    ERROR("Unknown spherical harmonics degree: {}", _sh_degree);
  }
  _scenemin = glm::min(_scenemin, _sceneminA);
  _scenemax = glm::max(_scenemax, _scenemaxA);

  // create mesh
  createMeshFromGaussians(pos, face, _mesh);

  // parse appearance gaussian data
  _appearancePoints.reserve(_gsCountA);
  for (int i = 0; i < _gsCountA; i++) {
    _appearancePoints.emplace_back(glm::vec3(posA[i].x, posA[i].y, posA[i].z),
                                   glm::vec3(coords[i].c[0], coords[i].c[1], coords[i].c[2]),
                                   static_cast<unsigned int>(faceIds[i]));
  }

  GaussianModel::_boxmin = _scenemin;
  GaussianModel::_boxmax = _scenemax;

  // Allocate and fill the GPU data
  cudaAllocCopy(&_pos_cuda, gsCount, pos, _gsCountA, posA);
  cudaAllocCopy(&_rot_cuda, gsCount, rot, _gsCountA, rotA);
  cudaAllocCopy(&_shs_cuda, gsCount, shs, _gsCountA, shsA);
  cudaAllocCopy(&_opacity_cuda, gsCount, opacity, _gsCountA, opacityA);
  cudaAllocCopy(&_scale_cuda, gsCount, scale, _gsCountA, scaleA);

  // Create space for view parameters
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_colmap_view_cuda, sizeof(glm::mat4)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_colmap_proj_view_cuda, sizeof(glm::mat4)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_cam_pos_cuda, 3 * sizeof(float)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_background_cuda, 3 * sizeof(float)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_rect_cuda, 2 * (gsCount + _gsCountA) * sizeof(int)));

  // Appearance face index
  std::vector<unsigned int> faceIds_uint;
  for (const float &f : faceIds) {
    faceIds_uint.push_back(static_cast<unsigned int>(f));
  }
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_appearance_face_idx_cuda, sizeof(unsigned int) * _gsCountA));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_appearance_face_idx_cuda, faceIds_uint.data(),
                                   sizeof(unsigned int) * _gsCountA, cudaMemcpyHostToDevice));
}

void TextureGaussianModel::initMesh() {

  _bvh.build(_mesh);

  _vertices.clear();
  _normal.clear();

  for (const MyMesh::FaceHandle &fh : _mesh.faces()) {
    for (const MyMesh::VertexHandle &vh : _mesh.fv_range(fh)) {

      _vertices.emplace_back(Utils::toGlm(_mesh.point(vh)));
      _normal.emplace_back(Utils::toGlm(_mesh.normal(vh)));
      _mesh.set_texcoord2D(vh, {0, 0});
    }
  }

  std::vector<glm::vec2> textureCoord = std::vector<glm::vec2>(n_vertices(), {0, 0});
  std::vector<cudaTextureObject_t> selectIdx(n_faces(), 0);
  std::vector<std::uint8_t> selectedFaces(n_faces(), 0);

  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_view_cuda, sizeof(glm::mat4)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_proj_cuda, sizeof(glm::mat4)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_mask_cuda, sizeof(CudaRasterizer::PixelMask) * pixels));
}

void TextureGaussianModel::updateTextureInfo(const TextureEditor &textureEditor) {

  size_t faceCount = _selectedID->size();
  auto selectedTexture = textureEditor.selectedPBR();
  std::vector<cudaTextureObject_t> basecolorIdx(
      faceCount, selectedTexture != nullptr ? selectedTexture->basecolor().cudaTextureId() : 0);
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_basecolor_map_cuda, basecolorIdx.data(),
                                   sizeof(cudaTextureObject_t) * faceCount, cudaMemcpyHostToDevice));
  std::vector<cudaTextureObject_t> normalIdx(
      faceCount, selectedTexture != nullptr ? selectedTexture->normal().cudaTextureId() : 0);
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_normal_map_cuda, normalIdx.data(),
                                   sizeof(cudaTextureObject_t) * faceCount, cudaMemcpyHostToDevice));
  std::vector<cudaTextureObject_t> heightIdx(
      faceCount, selectedTexture != nullptr ? selectedTexture->height().cudaTextureId() : 0);
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_height_map_cuda, heightIdx.data(),
                                   sizeof(cudaTextureObject_t) * faceCount, cudaMemcpyHostToDevice));
  std::vector<cudaTextureObject_t> roughnessIdx(
      faceCount, selectedTexture != nullptr ? selectedTexture->roughness().cudaTextureId() : 0);
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_roughness_map_cuda, roughnessIdx.data(),
                                   sizeof(cudaTextureObject_t) * faceCount, cudaMemcpyHostToDevice));
  std::vector<cudaTextureObject_t> maskIdx(
      faceCount, selectedTexture != nullptr ? selectedTexture->mask().cudaTextureId() : 0);
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_mask_filter_cuda, maskIdx.data(),
                                   sizeof(cudaTextureObject_t) * faceCount, cudaMemcpyHostToDevice));
}

void TextureGaussianModel::render(const Camera &camera, const int &width, const int &height,
                                  const glm::vec3 &clearColor, float *image_cuda,
                                  TextureEditor &textureEditor,
                                  CudaRasterizer::MaskCullingMode maskCullingMode,
                                  CudaRasterizer::Light light) {

  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_background_cuda, glm::value_ptr(clearColor), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  size_t pixels = (size_t)width * height;
  if (this->pixels != pixels) {
    cudaFree(_mask_cuda);
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_mask_cuda, sizeof(CudaRasterizer::PixelMask) * pixels));
  }

  // Compute additional view parameters
  float tan_fovy = std::tan(camera.fov() * 0.5f);
  float tan_fovx = tan_fovy * camera.aspect();

  // Copy frame-dependent data to GPU
  uploadColmapViewPorjMatrix(camera);
  CUDA_SAFE_CALL(
      cudaMemcpy(_view_cuda, glm::value_ptr(camera.viewMatrix()), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(_proj_cuda, glm::value_ptr(camera.projectionMatrix()), sizeof(glm::mat4),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(_cam_pos_cuda, glm::value_ptr(camera.eye()), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  // render selection
  size_t faceCount = _selectedID->size();
  auto selectedTexture = textureEditor.selectedPBR();
  CudaRasterizer::TextureOption textureOption{textureEditor.scale(), Utils::toFloat2(textureEditor.offset()),
                                              textureEditor.theta(), maskCullingMode};
  CUDA_SAFE_CALL(CudaRasterizer::makeMask(
      _model_position_cuda, _model_normal_cuda, _model_texCoords_cuda, _model_tangent_cuda,
      _model_bitangent_cuda, faceCount * 3, _model_basecolor_map_cuda, _model_normal_map_cuda,
      _model_height_map_cuda, _model_roughness_map_cuda, _model_mask_filter_cuda, textureOption,
      selectedTexture != nullptr ? selectedTexture->heightScale() : 0.0f, light, faceCount, _tessLevel, width,
      height, _view_cuda, _proj_cuda, _cam_pos_cuda, maskCullingMode, _mask_cuda));

  // Rasterize
  int *rects = _fastCulling ? _rect_cuda : nullptr;
  float *boxmin = _cropping ? glm::value_ptr(GaussianModel::_boxmin) : nullptr;
  float *boxmax = _cropping ? glm::value_ptr(GaussianModel::_boxmax) : nullptr;
  CudaRasterizer::forward(
      _geomBufferFunc, _binningBufferFunc, _imgBufferFunc, gsCount + _gsCountA, _sh_degree, MAX_SH_COEFF,
      _background_cuda, width, height, _pos_cuda, _shs_cuda, nullptr, _opacity_cuda, _scale_cuda,
      _scalingModifier, _rot_cuda, nullptr, _colmap_view_cuda, _colmap_proj_view_cuda, _cam_pos_cuda,
      tan_fovx, tan_fovy, false, image_cuda, _antialiasing, nullptr, rects, boxmin, boxmax, nullptr, nullptr,
      gsCount, _appearance_face_idx_cuda, _selected_face_idx_cuda, _selectedID->size(), _renderingMode,
      _mask_cuda, _threshold1, _threshold2, _threshold3, _threshold4, textureOption);

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error(std::format("A CUDA error occurred during rendering:{}. Please rerun "
                                         "in Debug to find the exact line!",
                                         cudaGetErrorString(cudaGetLastError())));
  }
}

void TextureGaussianModel::controls() {

  GaussianModel::controls();

  ImGui::NewLine();
  ImGui::Separator();
  ImGui::NewLine();

  ImGui::Combo("Rendering Mode", reinterpret_cast<int *>(&_renderingMode),
               Utils::enumToImGuiCombo<CudaRasterizer::RenderingMode>().c_str());

  ImGui::SliderFloat("threshold1", &_threshold1, 0.0f, 1.1f, "%.4f");
  ImGui::SliderFloat("threshold2", &_threshold2, 0.0f, 1.1f, "%.4f");
  ImGui::SliderFloat("threshold3", &_threshold3, 0.0f, 0.2f, "%.4f");
  ImGui::SliderFloat("threshold4", &_threshold4, 0.0f, 1.1f, "%.4f");

  ImGui::SliderInt("Tess Level", &_tessLevel, 1, 1024);
}

bool TextureGaussianModel::select(const glm::vec3 &hitPoint, int radius, bool isAdd) {

  bool dirty = Model::select(hitPoint, radius, isAdd);
  if (!dirty) {
    return false;
  }

  updateData();

  return true;
}

void TextureGaussianModel::updateData() {

  size_t faceCount = _selectedID->size();
  size_t vertexCount = faceCount * 3;

  std::vector<glm::vec3> v{};
  std::vector<glm::vec3> n{};
  std::vector<glm::vec2> t{};
  std::vector<glm::vec3> tangent{};
  std::vector<glm::vec3> bitangent{};

  for (const auto &fid : *_selectedID) {
    MyMesh::FaceHandle fh = _mesh.face_handle(fid);

    for (const MyMesh::VertexHandle vh : _mesh.fv_range(fh)) {

      v.push_back(Utils::toGlm(_mesh.point(vh)));
      n.push_back(Utils::toGlm(_mesh.normal(vh)));
      t.push_back(Utils::toGlm(_mesh.texcoord2D(vh)));
      tangent.push_back(Utils::toGlm(_mesh.data(vh).tangent));
      bitangent.push_back(Utils::toGlm(_mesh.data(vh).bitangent));
    }
  }

  // free old data
  cudaFree(_model_position_cuda);
  cudaFree(_selected_face_idx_cuda);
  cudaFree(_model_normal_cuda);
  cudaFree(_model_texCoords_cuda);
  cudaFree(_model_tangent_cuda);
  cudaFree(_model_bitangent_cuda);
  cudaFree(_model_basecolor_map_cuda);
  cudaFree(_model_normal_map_cuda);
  cudaFree(_model_height_map_cuda);
  cudaFree(_model_roughness_map_cuda);
  cudaFree(_model_mask_filter_cuda);

  // allocate new data
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_position_cuda, sizeof(glm::vec3) * vertexCount));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_model_position_cuda, v.data(), sizeof(glm::vec3) * vertexCount, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_normal_cuda, sizeof(glm::vec3) * vertexCount));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_model_normal_cuda, n.data(), sizeof(glm::vec3) * vertexCount, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_texCoords_cuda, sizeof(glm::vec2) * vertexCount));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_model_texCoords_cuda, t.data(), sizeof(glm::vec2) * vertexCount, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_tangent_cuda, sizeof(glm::vec3) * vertexCount));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_tangent_cuda, tangent.data(), sizeof(glm::vec3) * vertexCount,
                                   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_bitangent_cuda, sizeof(glm::vec3) * vertexCount));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_bitangent_cuda, bitangent.data(), sizeof(glm::vec3) * vertexCount,
                                   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMalloc((void **)&_model_basecolor_map_cuda, sizeof(cudaTextureObject_t) * faceCount));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMalloc((void **)&_model_normal_map_cuda, sizeof(cudaTextureObject_t) * faceCount));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMalloc((void **)&_model_height_map_cuda, sizeof(cudaTextureObject_t) * faceCount));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMalloc((void **)&_model_roughness_map_cuda, sizeof(cudaTextureObject_t) * faceCount));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMalloc((void **)&_model_mask_filter_cuda, sizeof(cudaTextureObject_t) * faceCount));

  std::vector<unsigned int> selectedIDData(_selectedID->begin(), _selectedID->end());
  CUDA_SAFE_CALL_ALWAYS(
      cudaMalloc((void **)&_selected_face_idx_cuda, sizeof(unsigned int) * _selectedID->size()));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_selected_face_idx_cuda, selectedIDData.data(),
                                   sizeof(unsigned int) * _selectedID->size(), cudaMemcpyHostToDevice));
}

void TextureGaussianModel::updateTexcoordVAO() {

  // update texcoord VAO buffer
  size_t faceCount = _selectedID->size();
  size_t vertexCount = faceCount * 3;
  auto *texcoordPtr = new glm::vec2[vertexCount];
  auto *tangentPtr = new glm::vec3[vertexCount];
  auto *bitangentPtr = new glm::vec3[vertexCount];

  // copy from CUDA
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(texcoordPtr, _model_texCoords_cuda, sizeof(glm::vec2) * vertexCount,
                                   cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(tangentPtr, _model_tangent_cuda, sizeof(glm::vec3) * vertexCount, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(bitangentPtr, _model_bitangent_cuda, sizeof(glm::vec3) * vertexCount,
                                   cudaMemcpyDeviceToHost));

  int index = 0;
  for (const auto &fid : *_selectedID) {
    MyMesh::FaceHandle fh = _mesh.face_handle(fid);

    for (const MyMesh::VertexHandle vh : _mesh.fv_range(fh)) {

      texcoordPtr[index] = Utils::toGlm(_mesh.texcoord2D(vh));
      tangentPtr[index] = Utils::toGlm(_mesh.data(vh).tangent);
      bitangentPtr[index] = Utils::toGlm(_mesh.data(vh).bitangent);
      index++;
    }
  }

  // copy to CUDA
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_texCoords_cuda, texcoordPtr, sizeof(glm::vec2) * vertexCount,
                                   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_model_tangent_cuda, tangentPtr, sizeof(glm::vec3) * vertexCount, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_bitangent_cuda, bitangentPtr, sizeof(glm::vec3) * vertexCount,
                                   cudaMemcpyHostToDevice));
}

void TextureGaussianModel::clearSelect() {

  Model::clearSelect();

  cudaFree(_model_position_cuda);
  _model_position_cuda = nullptr;
  cudaFree(_model_normal_cuda);
  _model_normal_cuda = nullptr;
  cudaFree(_model_texCoords_cuda);
  _model_texCoords_cuda = nullptr;
  cudaFree(_model_tangent_cuda);
  _model_tangent_cuda = nullptr;
  cudaFree(_model_bitangent_cuda);
  _model_bitangent_cuda = nullptr;
  cudaFree(_model_basecolor_map_cuda);
  _model_basecolor_map_cuda = nullptr;
  cudaFree(_model_normal_map_cuda);
  _model_normal_map_cuda = nullptr;
  cudaFree(_model_height_map_cuda);
  _model_height_map_cuda = nullptr;
  cudaFree(_model_roughness_map_cuda);
  _model_roughness_map_cuda = nullptr;
  cudaFree(_model_mask_filter_cuda);
  _model_mask_filter_cuda = nullptr;
  cudaFree(_selected_face_idx_cuda);
  _selected_face_idx_cuda = nullptr;
}

int TextureGaussianModel::count() const { return gsCount + _gsCountA; }

void TextureGaussianModel::solve(SolveUV::SolvingMode solvingMode, std::optional<glm::vec3> hitPoint) {

  Model::solve(solvingMode, hitPoint);

  for (AppearancePoint &point : _appearancePoints | std::ranges::views::filter([this](const auto &i) {
                                  return _selectedID->contains(i.faceId);
                                })) {

    MyMesh::FaceHandle fh = _mesh.face_handle(point.faceId);

    // get triangle vertex handle
    auto fv_it = _mesh.cfv_iter(fh);
    MyMesh::VertexHandle vh0 = *fv_it++;
    MyMesh::VertexHandle vh1 = *fv_it++;
    MyMesh::VertexHandle vh2 = *fv_it;

    // get point of vertices
    glm::vec3 v0 = Utils::toGlm(_mesh.point(vh0));
    glm::vec3 v1 = Utils::toGlm(_mesh.point(vh1));
    glm::vec3 v2 = Utils::toGlm(_mesh.point(vh2));

    glm::vec3 pVec = point.pos - v0;
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    float d00 = glm::dot(v0v1, v0v1);
    float d01 = glm::dot(v0v1, v0v2);
    float d11 = glm::dot(v0v2, v0v2);
    float d20 = glm::dot(pVec, v0v1);
    float d21 = glm::dot(pVec, v0v2);
    float denom = d00 * d11 - d01 * d01;

    // avoid to divide by zero
    if (std::abs(denom) > 1e-9f) {
      float v = (d11 * d20 - d01 * d21) / denom;
      float w = (d00 * d21 - d01 * d20) / denom;
      float u = 1.0f - v - w;

      point.uv = u * Utils::toGlm(_mesh.texcoord2D(vh0)) + v * Utils::toGlm(_mesh.texcoord2D(vh1)) +
                 w * Utils::toGlm(_mesh.texcoord2D(vh2));
    }
  }
}

std::vector<std::pair<unsigned int, std::pair<float, float>>>
TextureGaussianModel::getSelectedTextureCoords() const {

  auto result = Model::getSelectedTextureCoords();

  for (const AppearancePoint &point : _appearancePoints | std::ranges::views::filter([this](const auto &i) {
                                        return _selectedID->contains(i.faceId);
                                      })) {
    result.push_back({1, {point.uv.x, point.uv.y}});
  }

  return result;
}
