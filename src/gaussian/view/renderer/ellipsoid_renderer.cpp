#include "ellipsoid_renderer.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../utils/camera/camera.hpp"
#include "../utils/logger.hpp"
#include "model/gs_gl_data.hpp"

EllipsoidRenderer::EllipsoidRenderer() {

  program = std::make_unique<Program>(PROJECT_DIR "/src/gaussian/view/renderer/shader/ellipsoid.vert",
                                      PROJECT_DIR "/src/gaussian/view/renderer/shader/ellipsoid.frag", "");

  glCreateTextures(GL_TEXTURE_2D, 1, &idTexture);
  glTextureParameteri(idTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTextureParameteri(idTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glCreateTextures(GL_TEXTURE_2D, 1, &colorTexture);
  glTextureParameteri(colorTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTextureParameteri(colorTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glCreateFramebuffers(1, &fbo);
  glCreateRenderbuffers(1, &depthBuffer);

  makeFBO(800, 800);
}

EllipsoidRenderer::~EllipsoidRenderer() {
  glDeleteTextures(1, &idTexture);
  glDeleteTextures(1, &colorTexture);
  glDeleteRenderbuffers(1, &depthBuffer);
  glDeleteFramebuffers(1, &fbo);
}

void EllipsoidRenderer::makeFBO(int w, int h) {
  resX = w;
  resY = h;

  glBindTexture(GL_TEXTURE_2D, idTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, resX, resY, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

  glBindTexture(GL_TEXTURE_2D, colorTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resX, resY, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);

  glNamedRenderbufferStorage(depthBuffer, GL_DEPTH_COMPONENT, resX, resY);

  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, idTexture, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void EllipsoidRenderer::render(int G, const GaussianGLData &mesh, int width, int height, Camera &camera,
                               glm::vec3 clearColor, float alphaLimit) {

  if (width != resX || height != resY) {
    makeFBO(width, height);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, fbo);

  // Solid pass
  GLuint drawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, drawBuffers);

  glViewport(0, 0, width, height);

  glClearColor(clearColor.r, clearColor.g, clearColor.b, 1.0f);
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  program->use();
  program->setMat4("projection_matrix", camera.projectionMatrixPointer());
  program->setMat4("view_matrix", camera.viewMatrixPointer());
  auto model = glm::identity<glm::mat4>();
  program->setMat4("model_matrix", glm::value_ptr(model));
  program->setVec3("rayOrigin", glm::value_ptr(camera.eye()));
  program->setFloat("alpha_limit", alphaLimit);

  program->setInt("stage", 0);
  mesh.render(G);

  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) {
    ERROR("OpenGL Error after stage 0: {}", err);
  }

  // Simple additive blendnig (no order)
  glDrawBuffers(1, drawBuffers);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);
  glBlendEquation(GL_FUNC_ADD);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  program->setInt("stage", 1);
  mesh.render(G);

  while ((err = glGetError()) != GL_NO_ERROR) {
    ERROR("OpenGL Error after stage 1: {}", err);
  }

  glDepthMask(GL_TRUE);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  Program::unUse();

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
