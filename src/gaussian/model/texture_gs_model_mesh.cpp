#include "texture_gs_model_mesh.hpp"

#include <ImGui/imgui.h>

#include "gs_model.hpp"
#include "ply.hpp"
#include "rasterizer/defines.hpp"
#include "rasterizer/texture_rasterizer.hpp"
#include "utils/camera/camera.hpp"
#include "utils/utils.hpp"

#include "rasterizer/rasterizer.hpp"

TextureGaussianModelMesh::TextureGaussianModelMesh(const char *plyPath, const char *meshPath, int sh_degree,
                                                   int device)
    : GaussianModel(sh_degree, device), Model() {

  _loadPly(plyPath);
  loadMesh(meshPath);
}

TextureGaussianModelMesh::~TextureGaussianModelMesh() {

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
  cudaFree(_selected_face_idx_cuda);

  //
  cudaFree(_view_cuda);
  cudaFree(_proj_cuda);
  cudaFree(_mask_cuda);
}

void TextureGaussianModelMesh::initMesh() {

  _bvh.build(_mesh);

  _vertices.clear();
  _normal.clear();

  for (const MyMesh::FaceHandle &fh : _mesh.faces()) {
    for (const MyMesh::VertexHandle &vh : _mesh.fv_range(fh)) {

      _vertices.emplace_back(Utils::toGlm(_mesh.point(vh)));
      _normal.emplace_back(-Utils::toGlm(_mesh.normal(vh)));
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

void TextureGaussianModelMesh::updateTextureInfo(const TextureEditor &textureEditor) {

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

void TextureGaussianModelMesh::render(const Camera &camera, const int &width, const int &height,
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
  CudaRasterizer::forward(_geomBufferFunc, _binningBufferFunc, _imgBufferFunc, gsCount, _sh_degree,
                          MAX_SH_COEFF, _background_cuda, width, height, _pos_cuda, _shs_cuda, nullptr,
                          _opacity_cuda, _scale_cuda, _scalingModifier, _rot_cuda, nullptr, _colmap_view_cuda,
                          _colmap_proj_view_cuda, _cam_pos_cuda, tan_fovx, tan_fovy, false, image_cuda,
                          _antialiasing, nullptr, rects, boxmin, boxmax, nullptr, nullptr, -1, nullptr,
                          _selected_face_idx_cuda, _selectedID->size(), _renderingMode, _mask_cuda,
                          _threshold1, _threshold2, _threshold3, _threshold4, textureOption);

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error(std::format("A CUDA error occurred during rendering:{}. Please rerun "
                                         "in Debug to find the exact line!",
                                         cudaGetErrorString(cudaGetLastError())));
  }
}

void TextureGaussianModelMesh::controls() {

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

bool TextureGaussianModelMesh::select(const glm::vec3 &hitPoint, int radius, bool isAdd) {

  bool dirty = Model::select(hitPoint, radius, isAdd);
  if (!dirty) {
    return false;
  }

  updateData();

  return true;
}

void TextureGaussianModelMesh::updateData() {

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

void TextureGaussianModelMesh::updateTexcoordVAO() {

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

void TextureGaussianModelMesh::clearSelect() {

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

int TextureGaussianModelMesh::count() const { return gsCount; }
