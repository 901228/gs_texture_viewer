#ifndef POINT_RENDERER_HPP
#define POINT_RENDERER_HPP
#pragma once

#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../utils/gl/frameBufferHelper.hpp"
#include "../utils/gl/program.hpp"

class Camera;
class GaussianGLData;

class PointRenderer {

public:
  PointRenderer();

  void makeFBO(int width, int height);
  void render(int G, const GaussianGLData &mesh, int width, int height, Camera &camera, glm::vec3 clearColor,
              float alpha = 1.0f, int radius = 2, glm::vec3 user_color = {0.1f, 0.1f, 1.0f},
              glm::mat4 model = glm::identity<glm::mat4>());

  [[nodiscard]] unsigned int getTexture() const;

private:
  std::unique_ptr<Program> program;
  std::unique_ptr<FrameBufferHelper> fbo;
};

#endif // !POINT_RENDERER_HPP
