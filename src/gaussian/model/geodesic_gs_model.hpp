#ifndef GEODESIC_GS_MODEL_HPP
#define GEODESIC_GS_MODEL_HPP
#pragma once

#include "gs_model.hpp"

#include "utils/mesh/geodesic_splines.hpp"
#include "utils/texture/texture_editor.hpp"

#include "rasterizer/defines.hpp"

class GeodesicGaussianModel : public GaussianModel,
                              public TextureEditor::TextureEditableModel,
                              public GeodesicSplines::Implicit {
public:
  GeodesicGaussianModel(const char *plyPath, int sh_degree, int device = 0);
  ~GeodesicGaussianModel();

  bool resizeBuffer(int width, int height);
  void render(const Camera &camera, const int &width, const int &height, const glm::vec3 &clearColor,
              float *image_cuda, TextureEditor &textureEditor,
              CudaRasterizer::MaskCullingMode maskCullingMode, CudaRasterizer::Light light) const;
  void controls() override;

private:
  using GaussianModel::render;

  std::tuple<std::vector<Pos>, std::vector<Rot>, std::vector<Scale>, std::vector<float>>
  _loadPly(const char *plyPath) override;

  float *_inverse_colmap_view_cuda = nullptr;
  void uploadColmapViewPorjMatrix(const Camera &camera) const override;

private:
  int _lastW = 1, _lastH = 1;
  float *_pick_depth_cuda = nullptr; // raw blended depth (no bg compensation)
  float *_pick_T_cuda = nullptr;     // final_T

public:
  glm::vec3 _lastHitPos{};
  GeodesicSplines::LogarithmicMap::LogMapTable _logMap;
  std::vector<glm::vec3> _lastPoints;
  float _geodesicRadius = 0;
  std::optional<glm::vec3> hit(const Camera &camera, const glm::vec2 &ndcPos) const override;
  bool select(const glm::vec3 &hitPoint, int radius, bool isAdd) override;
  void clearSelect() override;
  void solve(SolveUV::SolvingMode solvingMode, std::optional<glm::vec3> hitPoint = std::nullopt) override;
  void updateTextureInfo(const TextureEditor &textureEditor) override;

private:
  // TODO: isosurface level, need to be tuned
  float threshold = 1.5f; // isosurface level, need to be tuned

  struct GaussianEntry {
    glm::vec3 pos;
    glm::mat3 invCov; // precomputed inverse covariance matrix (Σ^{-1})
    float alpha;
    float maxEigenvalue; // maximum eigenvalue, used for cutoff radius
  };
  std::vector<GaussianEntry> gaussians;

  // initialize from GaussianModel's host data
  constexpr static const int maxCellRange = 4;
  void buildGrid(const std::vector<Pos> &pos, const std::vector<Rot> &rot, const std::vector<Scale> &scale,
                 const std::vector<float> &opacity);

private:
  // spatial acceleration: 3D grid
  struct GridCell {
    std::vector<int> indices;
  };
  int gridRes = 16;
  float cellSize;
  glm::vec3 gridMin;
  std::vector<GridCell> grid;

  void _buildGrid();
  void queryNearby(const glm::vec3 &x, std::vector<int> &out) const;

public:
  const float eval(const glm::vec3 &x) override;
  const glm::vec3 grad(const glm::vec3 &x) override;
  const static int maxProjectIter = 20;
  const glm::vec3 project(const glm::vec3 &x) override;
  const glm::vec3 normal(const glm::vec3 &x) override;

private:
  float _threshold = 0.014f;

  cudaTextureObject_t _model_basecolor_map_cuda;
  cudaTextureObject_t _model_normal_map_cuda;
  cudaTextureObject_t _model_height_map_cuda;

  float *_last_points_cuda = nullptr;

  CudaRasterizer::PixelMask *_mask_cuda = nullptr;
  CudaRasterizer::RenderingMode _renderingMode = CudaRasterizer::RenderingMode::Color;
};

#endif // !GEODESIC_GS_MODEL_HPP
