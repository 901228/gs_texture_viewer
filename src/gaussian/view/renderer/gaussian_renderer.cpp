#include "gaussian_renderer.hpp"

#include "../utils/logger.hpp"

GaussianRenderer::GaussianRenderer() {

  copyProgram = std::make_unique<Program>(PROJECT_DIR "/src/gaussian/view/renderer/shader/copy.vert",
                                          PROJECT_DIR "/src/gaussian/view/renderer/shader/copy.frag", "");

  fbo = std::make_unique<FrameBufferHelper>(false, true);

  // Initialize screen quad
  {
    static const GLfloat Fvert[] = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
    static const GLfloat Ftcoord[] = {0, 0, 1, 0, 1, 1, 0, 1};
    static const GLuint Find[] = {0, 1, 2, 0, 2, 3};

    glGenVertexArrays(1, &vertexArrayObject);
    glBindVertexArray(vertexArrayObject);

    glGenBuffers(1, &indexVertexBufferObject);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVertexBufferObject);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Find), Find, GL_STATIC_DRAW);

    glGenBuffers(1, &vertexBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Fvert) + sizeof(Ftcoord), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Fvert), Fvert);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(Fvert), sizeof(Ftcoord), Ftcoord);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void *)sizeof(Fvert));

    glBindVertexArray(0);
  }
}

GaussianRenderer::~GaussianRenderer() {

  glBindVertexArray(vertexArrayObject);
  glDeleteBuffers(1, &vertexBufferObject);
  glDeleteBuffers(1, &indexVertexBufferObject);
  glBindVertexArray(0);

  glDeleteVertexArrays(1, &vertexArrayObject);
}

void GaussianRenderer::render(unsigned int bufferID, int width, int height, glm::vec3 clearColor,
                              bool disableTest, bool flip) {
  _copyBuffer(bufferID, width, height, clearColor, disableTest, flip);
}

void GaussianRenderer::_copyBuffer(unsigned int bufferID, int width, int height, glm::vec3 clearColor,
                                   bool disableTest, bool flip) {

  fbo->bindDraw();
  fbo->onResize(width, height);

  if (disableTest)
    glDisable(GL_DEPTH_TEST);
  else
    glEnable(GL_DEPTH_TEST);

  glClearColor(clearColor.r, clearColor.g, clearColor.b, 1.0f);
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  copyProgram->use();

  copyProgram->setInt("flip", flip);
  copyProgram->setInt("width", width);
  copyProgram->setInt("height", height);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

  glBindVertexArray(vertexArrayObject);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVertexBufferObject);

  const GLboolean cullingWasEnabled = glIsEnabled(GL_CULL_FACE);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

  if (!cullingWasEnabled) {
    glDisable(GL_CULL_FACE);
  }

  glBindVertexArray(0);

  glEnable(GL_DEPTH_TEST);

  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) {
    ERROR("OpenGL Error: {}", err);
  }

  Program::unUse();
  FrameBufferHelper::unbindDraw();
}

unsigned int GaussianRenderer::getTexture() const { return fbo->getTextureId(); }
