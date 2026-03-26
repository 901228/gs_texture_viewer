#ifndef ISOSURFACE_HPP
#define ISOSURFACE_HPP
#pragma once

#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "geodesic_splines.hpp"

class Program;
class Camera;

namespace Isosurface {

struct MarchingCubesResult {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<uint32_t> indices;
};

/**
 * Resolution Voxel數 速度  用途
 * 32         32K     即時  快速確認形狀
 * 64         262K    幾秒  開發期主力
 * 128        2M      慢    最終品質確認
 */
MarchingCubesResult extractIsosurface(GeodesicSplines::Implicit &model, glm::vec3 bmin, glm::vec3 bmax,
                                      int resolution = 64);

class IsosurfaceRenderer {
public:
  static inline IsosurfaceRenderer &getInstance() {
    static IsosurfaceRenderer instance;
    return instance;
  }
  IsosurfaceRenderer(IsosurfaceRenderer const &) = delete;
  void operator=(IsosurfaceRenderer const &) = delete;

private:
  IsosurfaceRenderer();
  ~IsosurfaceRenderer();

private:
  std::unique_ptr<Program> program;

  unsigned int vao = 0, vbo = 0, ebo = 0;
  int indexCount;

public:
  void upload(const MarchingCubesResult &mc);
  void render(const Camera &camera, const glm::vec3 &light);
};

} // namespace Isosurface

#endif // !ISOSURFACE_HPP
