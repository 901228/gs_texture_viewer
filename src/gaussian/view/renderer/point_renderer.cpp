#include "point_renderer.hpp"

#include "../utils/camera/camera.hpp"
#include "../utils/logger.hpp"
#include "model/gs_gl_data.hpp"

PointRenderer::PointRenderer() {

  program = std::make_unique<Program>(PROJECT_DIR "/src/gaussian/view/renderer/shader/alpha_points.vert",
                                      PROJECT_DIR "/src/gaussian/view/renderer/shader/alpha_points.frag", "");

  fbo = std::make_unique<FrameBufferHelper>(false, true);
}

void PointRenderer::render(int G, const GaussianGLData &mesh, int width, int height, Camera &camera,
                           glm::vec3 clearColor, float alpha, int radius, glm::vec3 user_color,
                           glm::mat4 model) {

  fbo->bindDraw();
  fbo->onResize(width, height);

  glClearColor(clearColor.r, clearColor.g, clearColor.b, 1.0f);
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glEnable(GL_PROGRAM_POINT_SIZE);
  program->use();

  program->setMat4("projection_matrix", camera.projectionMatrixPointer());
  program->setMat4("view_matrix", camera.viewMatrixPointer());
  program->setMat4("model_matrix", glm::value_ptr(model));
  program->setFloat("alpha", alpha);
  program->setInt("radius", radius);
  program->setVec3("user_color", glm::value_ptr(user_color));

  mesh.renderPoints(G);
  glDisable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_DEPTH_TEST);

  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) {
    ERROR("OpenGL Error: {}", err);
  }

  Program::unUse();
  FrameBufferHelper::unbindDraw();
}

unsigned int PointRenderer::getTexture() const { return fbo->getTextureId(); }
