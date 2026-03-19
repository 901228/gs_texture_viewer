#include "geodesic_gs_model.hpp"
#include "utils/logger.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/component_wise.hpp>

#include "ply.hpp"
#include "utils/mesh/solve_uv.hpp"

#include "rasterizer/rasterizer.hpp"

GeodesicGaussianModel::GeodesicGaussianModel(const char *plyPath, int sh_degree, int device)
    : GaussianModel(sh_degree, device) {

  _loadPly(plyPath);
}

GeodesicGaussianModel::~GeodesicGaussianModel() {}

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
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_pick_depth_cuda, sizeof(float) * _lastW * _lastH));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_pick_T_cuda, sizeof(float) * _lastW * _lastH));

  return true;
}

void GeodesicGaussianModel::render(const Camera &camera, const int &width, const int &height,
                                   const glm::vec3 &clearColor, float *image_cuda) const {

  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_background_cuda, glm::value_ptr(clearColor), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  // Compute additional view parameters
  float tan_fovy = std::tan(camera.fov() * 0.5f);
  float tan_fovx = tan_fovy * camera.aspect();

  // Copy frame-dependent data to GPU
  uploadColmapViewPorjMatrix(camera);
  CUDA_SAFE_CALL(
      cudaMemcpy(_cam_pos_cuda, glm::value_ptr(camera.eye()), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  // Rasterize
  int *rects = _fastCulling ? _rect_cuda : nullptr;
  float *boxmin = _cropping ? (float *)&_boxmin : nullptr;
  float *boxmax = _cropping ? (float *)&_boxmax : nullptr;
  CUDA_SAFE_CALL(CudaRasterizer::forward(_geomBufferFunc, _binningBufferFunc, _imgBufferFunc, gsCount,
                                         _sh_degree, MAX_SH_COEFF, _background_cuda, width, height, _pos_cuda,
                                         _shs_cuda, nullptr, _opacity_cuda, _scale_cuda, _scalingModifier,
                                         _rot_cuda, nullptr, _colmap_view_cuda, _colmap_proj_view_cuda,
                                         _cam_pos_cuda, tan_fovx, tan_fovy, false, image_cuda, _antialiasing,
                                         nullptr, rects, boxmin, boxmax, _pick_depth_cuda, _pick_T_cuda));

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error(std::format("A CUDA error occurred during rendering:{}. Please rerun "
                                         "in Debug to find the exact line!",
                                         cudaGetErrorString(cudaGetLastError())));
  }
}

void GeodesicGaussianModel::controls() { GaussianModel::controls(); }

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
  GeodesicSplines::debugStruct.center = hitPoint;
  return true;
}

void GeodesicGaussianModel::clearSelect() {}

void GeodesicGaussianModel::solve(SolveUV::SolvingMode solvingMode, std::optional<glm::vec3> hitPoint) {

  if (!hitPoint.has_value())
    return;

  SolveUV::SolveGeodesic(hitPoint.value(), *this);
}

void GeodesicGaussianModel::updateTextureInfo(const TextureEditor &textureEditor) {
  // TODO: update texture info
}

static glm::mat3 quatToMat(const Rot &r) {
  float w = r.rot[0], x = r.rot[1], y = r.rot[2], z = r.rot[3];
  return glm::mat3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 2 * (x * y - w * z),
                   1 - 2 * (x * x + z * z), 2 * (y * z + w * x), 2 * (x * z + w * y), 2 * (y * z - w * x),
                   1 - 2 * (x * x + y * y));
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
    glm::mat3 R = quatToMat(rot[k]);
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

  WARN("FAQ build grid");

  glm::vec3 bmin(1e9f), bmax(-1e9f);
  for (auto &g : gaussians) {
    bmin = glm::min(bmin, g.pos - g.maxEigenvalue * 3.f);
    bmax = glm::max(bmax, g.pos + g.maxEigenvalue * 3.f);
  }
  gridMin = bmin - 1e-4f;
  cellSize = glm::compMax(bmax - bmin) / gridRes;
  grid.assign(gridRes * gridRes * gridRes, {});

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
    xi -= (f / gf2) * gf;

    if (std::abs(f) < 1e-5f)
      break;
  }
  return xi;
}

const glm::vec3 GeodesicGaussianModel::normal(const glm::vec3 &x) {

  glm::vec3 g = grad(x);
  float len = glm::length(g);
  if (len < 1e-8f)
    return glm::vec3(0, 1, 0);
  // f = σ - threshold，∇f 指向 density 增加方向（朝內）
  // 表面法線朝外，所以取負號
  return -g / len;
}
