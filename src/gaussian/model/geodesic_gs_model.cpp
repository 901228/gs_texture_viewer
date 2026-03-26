#include "geodesic_gs_model.hpp"
#include "imgui.h"
#include "utils/mesh/geodesic_splines.hpp"
#include <cuda_runtime_api.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/component_wise.hpp>

#include "ply.hpp"
#include "utils/mesh/solve_uv.hpp"
#include "utils/utils.hpp"

#include "rasterizer/geodesics.hpp"
#include "rasterizer/rasterizer.hpp"

GeodesicGaussianModel::GeodesicGaussianModel(const char *plyPath, int sh_degree, int device)
    : GaussianModel(sh_degree, device) {

  _loadPly(plyPath);

  CUDA_SAFE_CALL_ALWAYS(
      cudaMalloc((void **)&_mask_cuda, sizeof(CudaRasterizer::PixelMask) * _lastW * _lastH));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_inverse_colmap_view_cuda, sizeof(glm::mat4)));
}

GeodesicGaussianModel::~GeodesicGaussianModel() {

  cudaFree(_last_points_cuda);
  cudaFree(_mask_cuda);
  cudaFree(_colmap_proj_view_cuda);
}

std::tuple<std::vector<Pos>, std::vector<Rot>, std::vector<Scale>, std::vector<float>>
GeodesicGaussianModel::_loadPly(const char *plyPath) {

  auto res = GaussianModel::_loadPly(plyPath);
  const auto &[pos, rot, scale, opacity] = res;

  buildGrid(pos, rot, scale, opacity);

  return res;
}

bool GeodesicGaussianModel::resizeBuffer(int width, int height) {

  if (_lastW == width && _lastH == height)
    return false;

  _lastW = width;
  _lastH = height;

  cudaFree(_pick_depth_cuda);
  cudaFree(_pick_T_cuda);
  cudaFree(_mask_cuda);
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_pick_depth_cuda, sizeof(float) * _lastW * _lastH));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_pick_T_cuda, sizeof(float) * _lastW * _lastH));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMalloc((void **)&_mask_cuda, sizeof(CudaRasterizer::PixelMask) * _lastW * _lastH));

  return true;
}

void GeodesicGaussianModel::uploadColmapViewPorjMatrix(const Camera &camera) const {

  // Convert view and projection to target coordinate system
  glm::mat4 view_mat{camera.viewMatrix()};
  glm::mat4 proj_view_mat = camera.projectionMatrix() * view_mat;
  flipRow(view_mat, 1);
  flipRow(view_mat, 2);
  flipRow(proj_view_mat, 1);
  glm::mat4 inverse_view_mat = glm::inverse(view_mat);

  CUDA_SAFE_CALL(
      cudaMemcpy(_colmap_view_cuda, glm::value_ptr(view_mat), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(_colmap_proj_view_cuda, glm::value_ptr(proj_view_mat), sizeof(glm::mat4),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(_inverse_colmap_view_cuda, glm::value_ptr(inverse_view_mat), sizeof(glm::mat4),
                            cudaMemcpyHostToDevice));
}

void GeodesicGaussianModel::render(const Camera &camera, const int &width, const int &height,
                                   const glm::vec3 &clearColor, float *image_cuda,
                                   TextureEditor &textureEditor,
                                   CudaRasterizer::MaskCullingMode maskCullingMode,
                                   CudaRasterizer::Light light) const {

  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_background_cuda, glm::value_ptr(clearColor), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  // Compute additional view parameters
  float tan_fovy = std::tan(camera.fov() * 0.5f);
  float tan_fovx = tan_fovy * camera.aspect();

  // Copy frame-dependent data to GPU
  uploadColmapViewPorjMatrix(camera);
  CUDA_SAFE_CALL(
      cudaMemcpy(_cam_pos_cuda, glm::value_ptr(camera.eye()), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  // render selection
  CUDA_SAFE_CALL(CudaRasterizer::makeGeodesicsMask(
      _pick_depth_cuda, _pick_T_cuda, _logMap.pts3d_cuda(), _logMap.nPts(), _logMap.uvs_cuda(),
      _logMap.gridData_cuda(), _logMap.gridOffsets_cuda(), _logMap.gridRes(),
      {_logMap.gridMin().x, _logMap.gridMin().y, _logMap.gridMin().z}, _logMap.cellSize(), _geodesicRadius,
      _last_points_cuda, GeodesicSplines::settings.m, _colmap_proj_view_cuda, width, height,
      _inverse_colmap_view_cuda, tan_fovx, tan_fovy, _model_basecolor_map_cuda, _model_normal_map_cuda,
      _model_height_map_cuda, _mask_cuda));

  // Rasterize
  int *rects = _fastCulling ? _rect_cuda : nullptr;
  float *boxmin = _cropping ? (float *)&_boxmin : nullptr;
  float *boxmax = _cropping ? (float *)&_boxmax : nullptr;
  CudaRasterizer::TextureOption textureOption{textureEditor.scale(), Utils::toFloat2(textureEditor.offset()),
                                              textureEditor.theta(), maskCullingMode};
  CUDA_SAFE_CALL(CudaRasterizer::forward(
      _geomBufferFunc, _binningBufferFunc, _imgBufferFunc, gsCount, _sh_degree, MAX_SH_COEFF,
      _background_cuda, width, height, _pos_cuda, _shs_cuda, nullptr, _opacity_cuda, _scale_cuda,
      _scalingModifier, _rot_cuda, nullptr, _colmap_view_cuda, _colmap_proj_view_cuda, _cam_pos_cuda,
      tan_fovx, tan_fovy, false, image_cuda, _antialiasing, nullptr, rects, boxmin, boxmax, _pick_depth_cuda,
      _pick_T_cuda, _renderingMode, _mask_cuda, _threshold, textureOption));

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error(std::format("A CUDA error occurred during rendering:{}. Please rerun "
                                         "in Debug to find the exact line!",
                                         cudaGetErrorString(cudaGetLastError())));
  }
}

void GeodesicGaussianModel::controls() {

  ImGui::SliderFloat("isosurface threshold", &threshold, 1.0f, 10.0f);
  ImGui::NewLine();

  GaussianModel::controls();

  ImGui::NewLine();
  ImGui::Separator();
  ImGui::NewLine();

  ImGui::Combo("Rendering Mode", reinterpret_cast<int *>(&_renderingMode),
               Utils::enumToImGuiCombo<CudaRasterizer::RenderingMode>().c_str());

  ImGui::SliderFloat("threshold", &_threshold, 0.0f, 0.02f, "%.4f");
}

std::optional<glm::vec3> GeodesicGaussianModel::hit(const Camera &camera, const glm::vec2 &ndcPos) const {

  glm::vec2 p =
      glm::clamp(Camera::getlocalPosFromNDC(ndcPos, _lastW, _lastH), {0, 0}, {_lastW - 1, _lastH - 1});
  int pix_id = static_cast<int>(p.y) * _lastW + static_cast<int>(p.x);

  float T_val, depth_val;
  cudaMemcpy(&T_val, _pick_T_cuda + pix_id, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&depth_val, _pick_depth_cuda + pix_id, sizeof(float), cudaMemcpyDeviceToHost);

  if ((1.0f - T_val) < 0.5f)
    return std::nullopt;

  float tan_fovy = std::tan(camera.fov() * 0.5f);
  float tan_fovx = tan_fovy * camera.aspect();

  glm::vec3 viewspace_pos{ndcPos.x * depth_val * tan_fovx,
                          -ndcPos.y * depth_val * tan_fovy, // flip Y axis
                          depth_val};

  // colmap view matrix
  glm::mat4 colmap_view = camera.viewMatrix();
  GaussianModel::flipRow(colmap_view, 1);
  GaussianModel::flipRow(colmap_view, 2);

  // pos_world = inv(colmap_view) * viewspace_pos
  glm::vec4 pos_world = glm::inverse(colmap_view) * glm::vec4(viewspace_pos, 1.0f);

  return glm::vec3(pos_world);
}

bool GeodesicGaussianModel::select(const glm::vec3 &hitPoint, int radius, bool isAdd) {

  if (_lastHitPos == hitPoint)
    return false;

  _lastHitPos = hitPoint;
  return true;
}

void GeodesicGaussianModel::clearSelect() {}

void GeodesicGaussianModel::solve(SolveUV::SolvingMode solvingMode, std::optional<glm::vec3> hitPoint) {

  if (!hitPoint.has_value())
    return;

  std::tie(_logMap, _lastPoints, _geodesicRadius) = SolveUV::SolveGeodesic(hitPoint.value(), *this);
  _logMap.upload();

  cudaFree(_last_points_cuda);
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_last_points_cuda, sizeof(glm::vec3) * _lastPoints.size()));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_last_points_cuda, _lastPoints.data(),
                                   sizeof(glm::vec3) * _lastPoints.size(), cudaMemcpyHostToDevice));
}

void GeodesicGaussianModel::updateTextureInfo(const TextureEditor &textureEditor) {

  auto selectedTexture = textureEditor.selectedPBR();
  _model_basecolor_map_cuda = selectedTexture != nullptr ? selectedTexture->basecolor().cudaTextureId() : 0;
  _model_normal_map_cuda = selectedTexture != nullptr ? selectedTexture->normal().cudaTextureId() : 0;
  _model_height_map_cuda = selectedTexture != nullptr ? selectedTexture->height().cudaTextureId() : 0;
}

void GeodesicGaussianModel::buildGrid(const std::vector<Pos> &pos, const std::vector<Rot> &rot,
                                      const std::vector<Scale> &scale, const std::vector<float> &opacity) {

  gaussians.clear();
  gaussians.reserve(pos.size());

  for (int k = 0; k < (int)pos.size(); ++k) {
    GaussianEntry e;
    e.pos = {pos[k].x, pos[k].y, pos[k].z};
    e.alpha = opacity[k];

    // Σ = R * S² * R^T，Σ^{-1} = R * S^{-2} * R^T
    glm::mat3 R = glm::mat3(glm::mat4(glm::quat(rot[k].rot[0], rot[k].rot[1], rot[k].rot[2], rot[k].rot[3])));
    glm::vec3 s = {scale[k].scale[0], scale[k].scale[1], scale[k].scale[2]};

    // S^{-2} diagonal
    glm::mat3 invS2 = glm::mat3(0);
    invS2[0][0] = 1.f / (s.x * s.x);
    invS2[1][1] = 1.f / (s.y * s.y);
    invS2[2][2] = 1.f / (s.z * s.z);

    // Σ^{-1} = R * invS2 * R^T
    e.invCov = R * invS2 * glm::transpose(R);

    // cutoff: 3σ of largest axis
    e.maxEigenvalue = glm::max(s.x, glm::max(s.y, s.z));

    gaussians.push_back(e);
  }

  _buildGrid();
}

void GeodesicGaussianModel::_buildGrid() {

  glm::vec3 bmin(1e9f), bmax(-1e9f);
  for (auto &g : gaussians) {
    bmin = glm::min(bmin, g.pos - g.maxEigenvalue * 3.f);
    bmax = glm::max(bmax, g.pos + g.maxEigenvalue * 3.f);
  }
  gridMin = bmin - 1e-4f;
  cellSize = glm::compMax(bmax - bmin) / gridRes;

  int nCells = gridRes * gridRes * gridRes;
  grid.assign(nCells, {});

  for (int k = 0; k < (int)gaussians.size(); ++k) {
    const auto &g = gaussians[k];

    // calculate the cell range covered by this Gaussian (3σ radius)
    glm::ivec3 center =
        glm::clamp(glm::ivec3((g.pos - gridMin) / cellSize), glm::ivec3(0), glm::ivec3(gridRes - 1));

    int r = std::min(static_cast<int>(std::ceil(g.maxEigenvalue * 3.f / cellSize)), maxCellRange);
    for (int dz = -r; dz <= r; ++dz) {
      for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
          glm::ivec3 nb = center + glm::ivec3(dx, dy, dz);
          if (glm::any(glm::lessThan(nb, glm::ivec3(0))))
            continue;
          if (glm::any(glm::greaterThanEqual(nb, glm::ivec3(gridRes))))
            continue;
          int idx = nb.x + gridRes * (nb.y + gridRes * nb.z);
          grid[idx].indices.push_back(k);
        }
      }
    }
  }
}

void GeodesicGaussianModel::queryNearby(const glm::vec3 &x, std::vector<int> &out) const {

  out.clear();
  glm::ivec3 cell = glm::clamp(glm::ivec3((x - gridMin) / cellSize), glm::ivec3(0), glm::ivec3(gridRes - 1));

  int idx = cell.x + gridRes * (cell.y + gridRes * cell.z);
  out = grid[idx].indices;
}

const float GeodesicGaussianModel::eval(const glm::vec3 &x) {

  float sigma = 0.f;
  // only accelerate querying nearby Gaussians
  std::vector<int> nearby;
  queryNearby(x, nearby);

  for (int k : nearby) {
    const auto &g = gaussians[k];
    glm::vec3 d = x - g.pos;
    float exponent = -0.5f * glm::dot(d, g.invCov * d);
    if (exponent < -9.f)
      continue; // exp < e^{-9} ≈ 0
    sigma += g.alpha * std::exp(exponent);
  }
  return sigma - threshold;
}

const glm::vec3 GeodesicGaussianModel::grad(const glm::vec3 &x) {

  glm::vec3 g(0);
  std::vector<int> nearby;
  queryNearby(x, nearby);

  for (int k : nearby) {
    const auto &gs = gaussians[k];
    glm::vec3 d = x - gs.pos;
    float exponent = -0.5f * glm::dot(d, gs.invCov * d);
    if (exponent < -9.f)
      continue;
    float w = gs.alpha * std::exp(exponent);
    // ∇_x exp(...) = exp(...) * (-Σ^{-1} d)
    g += w * (-gs.invCov * d);
  }
  return g;
}

const glm::vec3 GeodesicGaussianModel::project(const glm::vec3 &x) {

  glm::vec3 xi = x;
  for (int i = 0; i < maxProjectIter; ++i) {
    float f = eval(xi);
    glm::vec3 gf = grad(xi);
    float gf2 = glm::dot(gf, gf);
    if (gf2 < 1e-10f)
      break;

    // Newton step: x_{i+1} = x_i - f(x_i)/||∇f||² * ∇f
    float step = glm::clamp(f / gf2, -0.05f, 0.05f);
    xi -= step * gf;

    if (std::abs(f) < 1e-5f)
      break;
  }
  return xi;
}

const glm::vec3 GeodesicGaussianModel::normal(const glm::vec3 &x) {

  // glm::vec3 g = grad(x);
  // float len = glm::length(g);

  // if (len < 1e-8f) {
  //   // fallback: use the direction towards the center of the nearest Gaussian
  //   std::vector<int> nearby;
  //   queryNearby(x, nearby);
  //   float bestDist = 1e9f;
  //   glm::vec3 bestDir(0, 1, 0);
  //   for (int k : nearby) {
  //     glm::vec3 d = x - gaussians[k].pos;
  //     float dist = glm::length(d);
  //     if (dist < bestDist && dist > 1e-6f) {
  //       bestDist = dist;
  //       bestDir = d / dist;
  //     }
  //   }
  //   return bestDir;
  // }

  // // f = σ - threshold，∇f is the direction towards density increase (towards inside)
  // // surface normal points outwards, so take negative
  // return -g / len;

  float eps = cellSize * 0.5f; // set eps to cellSize / 2

  // use finite differences to compute the gradient
  float dx = eval(x + glm::vec3(eps, 0, 0)) - eval(x - glm::vec3(eps, 0, 0));
  float dy = eval(x + glm::vec3(0, eps, 0)) - eval(x - glm::vec3(0, eps, 0));
  float dz = eval(x + glm::vec3(0, 0, eps)) - eval(x - glm::vec3(0, 0, eps));

  glm::vec3 g(dx, dy, dz);
  float len = glm::length(g);
  if (len < 1e-8f)
    return glm::vec3(0, 1, 0);
  return -g / len;
}
