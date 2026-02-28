
#ifndef ELLIPSOID_RENDERER_HPP
#define ELLIPSOID_RENDERER_HPP
#pragma once

#include <glm/glm.hpp>
#include <memory>

#include "../utils/gl/program.hpp"

class Camera;
class GaussianGLData;

class EllipsoidRenderer {
public:
  EllipsoidRenderer();
  ~EllipsoidRenderer();

  void makeFBO(int width, int height);
  void render(int G, const GaussianGLData &mesh, int width, int height, Camera &camera, glm::vec3 clearColor,
              float alphaLimit = 0.2f);

private:
  std::unique_ptr<Program> program;

  unsigned int idTexture = 0;
  unsigned int colorTexture = 0;
  unsigned int depthBuffer = 0;
  unsigned int fbo = 0;
  int resX = 800, resY = 800;

public:
  [[nodiscard]] inline unsigned int getTexture() const { return colorTexture; }
};

#endif // !ELLIPSOID_RENDERER_HPP
