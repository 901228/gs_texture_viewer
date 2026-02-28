#include "texture_gs_model.hpp"

#include <ranges>
#include <utility>

#include "../utils/camera/camera.hpp"
#include "../utils/utils.hpp"
#include "gs_model.hpp"
#include "imgui.h"
#include "ply.hpp"
#include "utils.hpp"

#include "rasterizer/rasterizer.hpp"

TextureGaussianModel::TextureGaussianModel(const char *geometryPlyPath, const char *appearancePlyPath,
                                           int sh_degree, int device)
    : GaussianModel(sh_degree, device), Model() {

  _loadPly(geometryPlyPath, appearancePlyPath);
  initMesh();
  initModelForCuda();
}

TextureGaussianModel::~TextureGaussianModel() {

  //
  cudaFree(_model_view_cuda);
  cudaFree(_model_proj_cuda);
  cudaFree(_model_position_cuda);
  cudaFree(_model_texCoords_cuda);
  cudaFree(_model_sl_cuda);
  cudaFree(_model_face_mask_cuda);

  //
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

  _boxmin = _scenemin;
  _boxmax = _scenemax;

  // Allocate and fill the GPU data
  cudaAllocCopy(&_pos_cuda, gsCount, pos, _gsCountA, posA);
  cudaAllocCopy(&_rot_cuda, gsCount, rot, _gsCountA, rotA);
  cudaAllocCopy(&_shs_cuda, gsCount, shs, _gsCountA, shsA);
  cudaAllocCopy(&_opacity_cuda, gsCount, opacity, _gsCountA, opacityA);
  cudaAllocCopy(&_scale_cuda, gsCount, scale, _gsCountA, scaleA);

  // Create space for view parameters
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_view_cuda, sizeof(glm::mat4)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_proj_cuda, sizeof(glm::mat4)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_cam_pos_cuda, 3 * sizeof(float)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_background_cuda, 3 * sizeof(float)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_rect_cuda, 2 * (gsCount + _gsCountA) * sizeof(int)));
}

void TextureGaussianModel::initModelForCuda() {

  std::vector<glm::vec2> textureCoord = std::vector<glm::vec2>(n_vertices(), {0, 0});
  std::vector<cudaTextureObject_t> selectIdx(n_faces(), 0);
  std::vector<std::uint8_t> selectedFaces(n_faces(), 0);

  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_position_cuda, sizeof(glm::vec3) * n_vertices()));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_position_cuda, vertices().data(), sizeof(glm::vec3) * n_vertices(),
                                   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_texCoords_cuda, sizeof(glm::vec2) * n_vertices()));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_texCoords_cuda, textureCoord.data(),
                                   sizeof(glm::vec2) * n_vertices(), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_sl_cuda, sizeof(cudaTextureObject_t) * n_faces()));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_sl_cuda, selectIdx.data(), sizeof(cudaTextureObject_t) * n_faces(),
                                   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_face_mask_cuda, sizeof(std::uint8_t) * n_faces()));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_face_mask_cuda, selectedFaces.data(),
                                   sizeof(std::uint8_t) * n_faces(), cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_view_cuda, sizeof(glm::mat4)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_model_proj_cuda, sizeof(glm::mat4)));

  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_mask_cuda, sizeof(CudaRasterizer::PixelMask) * pixels));
}

void TextureGaussianModel::render(const Camera &camera, const int &width, const int &height,
                                  const glm::vec3 &clearColor, float *image_cuda, cudaTextureObject_t texId,
                                  const CudaRasterizer::TextureOption &textureOption) {

  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_background_cuda, glm::value_ptr(clearColor), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  size_t pixels = (size_t)width * height;
  if (this->pixels != pixels) {
    cudaFree(_mask_cuda);
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_mask_cuda, sizeof(CudaRasterizer::PixelMask) * pixels));
  }

  // Convert view and projection to target coordinate system
  glm::mat4 view_mat{camera.viewMatrix()};
  glm::mat4 proj_view_mat = camera.projectionMatrix() * view_mat;
  Utils::flipRow(view_mat, 1);
  Utils::flipRow(view_mat, 2);
  Utils::flipRow(proj_view_mat, 1);

  // Compute additional view parameters
  float tan_fovy = std::tan(camera.fov() * 0.5f);
  float tan_fovx = tan_fovy * camera.aspect();

  // Copy frame-dependent data to GPU
  CUDA_SAFE_CALL(cudaMemcpy(_view_cuda, glm::value_ptr(view_mat), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(_proj_cuda, glm::value_ptr(proj_view_mat), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(_cam_pos_cuda, glm::value_ptr(camera.eye()), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  // render selection
  std::vector<cudaTextureObject_t> selectIdx(n_faces(), texId);
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_sl_cuda, selectIdx.data(), sizeof(cudaTextureObject_t) * n_faces(),
                                   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(_model_view_cuda, glm::value_ptr(camera.viewMatrix()), sizeof(glm::mat4),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(_model_proj_cuda, glm::value_ptr(camera.projectionMatrix()), sizeof(glm::mat4),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(CudaRasterizer::makeMask(_model_position_cuda, _model_texCoords_cuda, n_vertices(),
                                          _model_sl_cuda, n_faces(), _model_face_mask_cuda, width, height,
                                          _model_view_cuda, _model_proj_cuda, _mask_cuda));

  // Rasterize
  int *rects = _fastCulling ? _rect_cuda : nullptr;
  float *boxmin = _cropping ? (float *)&_boxmin : nullptr;
  float *boxmax = _cropping ? (float *)&_boxmax : nullptr;
  CudaRasterizer::forward(
      _geomBufferFunc, _binningBufferFunc, _imgBufferFunc, gsCount + _gsCountA, _sh_degree, MAX_SH_COEFF,
      _background_cuda, width, height, _pos_cuda, _shs_cuda, nullptr, _opacity_cuda, _scale_cuda,
      _scalingModifier, _rot_cuda, nullptr, _view_cuda, _proj_cuda, _cam_pos_cuda, tan_fovx, tan_fovy, false,
      image_cuda, _antialiasing, nullptr, rects, boxmin, boxmax, _mask_cuda, _threshold, textureOption);

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error(std::format("A CUDA error occurred during rendering:{}. Please rerun "
                                         "in Debug to find the exact line!",
                                         cudaGetErrorString(cudaGetLastError())));
  }
}

void TextureGaussianModel::controls() {

  GaussianModel::controls();

  ImGui::SliderFloat("threshold", &_threshold, 0.0f, 0.005f, "%.4f");
}

void TextureGaussianModel::selectRadius(int id, int radius, bool isAdd) {

  Model::selectRadius(id, radius, isAdd);

  std::vector<std::uint8_t> selectedFaces(n_faces(), 0);
  for (const auto &i : *_selectedID) {
    selectedFaces[i] = 1;
  }
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_face_mask_cuda, selectedFaces.data(),
                                   sizeof(std::uint8_t) * n_faces(), cudaMemcpyHostToDevice));
}

glm::vec2 *TextureGaussianModel::updateTexcoordVAO(bool returnData) {

  auto data = Model::updateTexcoordVAO(true);

  // copy to CUDA
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_model_texCoords_cuda, data, sizeof(glm::vec2) * n_vertices(), cudaMemcpyHostToDevice));

  if (returnData) {
    return data;
  } else {
    delete[] data;
    return nullptr;
  };
}

void TextureGaussianModel::clearSelect() {

  Model::clearSelect();

  std::vector<std::uint8_t> selectedFaces(n_faces(), 0);
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_model_face_mask_cuda, selectedFaces.data(),
                                   sizeof(std::uint8_t) * n_faces(), cudaMemcpyHostToDevice));
}

int TextureGaussianModel::count() const { return gsCount + _gsCountA; }

void TextureGaussianModel::calculateParameterization(SolveUV::SolvingMode solvingMode, float angle) {

  Model::calculateParameterization(solvingMode, angle);

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
