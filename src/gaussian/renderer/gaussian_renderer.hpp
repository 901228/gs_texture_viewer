#ifndef GAUSSIAN_RENDERER_HPP
#define GAUSSIAN_RENDERER_HPP
#pragma once

#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../utils/gl/frameBufferHelper.hpp"
#include "../utils/gl/program.hpp"

class GaussianRenderer {

public:
  GaussianRenderer();
  ~GaussianRenderer();

  void makeFBO(int width, int height);
  void render(unsigned int bufferID, int width, int height, glm::vec3 clearColor, bool disableTest = true,
              bool flip = false);

  [[nodiscard]] unsigned int getTexture() const;

private:
  // void _renderGaussian();
  void _copyBuffer(unsigned int bufferID, int width, int height, glm::vec3 clearColor,
                   bool disableTest = true, bool flip = false);

private:
  std::unique_ptr<Program> copyProgram;
  std::unique_ptr<FrameBufferHelper> fbo;

  unsigned int vertexArrayObject = 0;
  unsigned int indexVertexBufferObject = 0;
  unsigned int vertexBufferObject = 0;
};

#endif // !GAUSSIAN_RENDERER_HPP
