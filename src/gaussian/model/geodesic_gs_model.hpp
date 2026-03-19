#ifndef GEODESIC_GS_MODEL_HPP
#define GEODESIC_GS_MODEL_HPP
#pragma once

#include "gs_model.hpp"

#include "ply.hpp"
#include "utils/mesh/geodesic_splines.hpp"
#include "utils/texture/texture_editor.hpp"

class GeodesicGaussianModel : public GaussianModel,
                              public TextureEditor::TextureEditableModel,
                              public GeodesicSplines::Implicit {
public:
  GeodesicGaussianModel(const char *plyPath, int sh_degree, int device = 0);
  ~GeodesicGaussianModel();

  bool resizeBuffer(int width, int height);
  void render(const Camera &camera, const int &width, const int &height, const glm::vec3 &clearColor,
              float *image_cuda) const override;
  void controls() override;

private:
  std::tuple<std::vector<Pos>, std::vector<Rot>, std::vector<Scale>, std::vector<float>>
  _loadPly(const char *plyPath) override;

private:
  int _lastW = 1, _lastH = 1;
  float *_pick_depth_cuda = nullptr; // raw blended depth (no bg compensation)
  float *_pick_T_cuda = nullptr;     // final_T

public:
  glm::vec3 _lastHitPos{};
  std::optional<glm::vec3> hit(const Camera &camera, const glm::vec2 &ndcPos) const override;
  bool select(const glm::vec3 &hitPoint, int radius, bool isAdd) override;
  void clearSelect() override;
  void solve(SolveUV::SolvingMode solvingMode, std::optional<glm::vec3> hitPoint = std::nullopt) override;
  void updateTextureInfo(const TextureEditor &textureEditor) override;

private:
  struct GaussianEntry {
    glm::vec3 pos;
    glm::mat3 invCov; // precomputed inverse covariance matrix (Σ^{-1})
    float alpha;
    float maxEigenvalue; // maximum eigenvalue, used for cutoff radius
  };

  std::vector<GaussianEntry> gaussians;
  float threshold = 0.3f; // isosurface level, need to be tuned

  // initialize from GaussianModel's host data
  void buildGrid(const std::vector<Pos> &pos, const std::vector<Rot> &rot, const std::vector<Scale> &scale,
                 const std::vector<float> &opacity);

private:
  // spatial acceleration: 3D grid
  struct GridCell {
    std::vector<int> indices;
  };
  int gridRes = 32;
  float cellSize;
  glm::vec3 gridMin;
  std::vector<GridCell> grid;

  constexpr static const int maxCellRange = 4;
  void _buildGrid();
  void queryNearby(const glm::vec3 &x, std::vector<int> &out) const;

public:
  const float eval(const glm::vec3 &x);
  const glm::vec3 grad(const glm::vec3 &x);
  const static int maxProjectIter = 20;
  const glm::vec3 project(const glm::vec3 &x) override;
  const glm::vec3 normal(const glm::vec3 &x) override;
};

#endif // !GEODESIC_GS_MODEL_HPP
