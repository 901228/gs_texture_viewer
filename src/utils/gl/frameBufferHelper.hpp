#ifndef FRAME_BUFFER_HELPER_HPP
#define FRAME_BUFFER_HELPER_HPP
#pragma once

#include <glad/gl.h>

#include "../logger.hpp"

class FrameBufferHelper {

public:
  inline explicit FrameBufferHelper(bool isSelect = false, bool hasAlpha = false)
      : isSelect(isSelect), hasAlpha(hasAlpha) {
    this->create();
  }

  inline ~FrameBufferHelper() {
    glDeleteTextures(1, &texture_id);
    glDeleteRenderbuffers(1, &renderBufferObject);
    glDeleteFramebuffers(1, &frameBufferObject);
  }

public:
  inline void create() {

    glGenFramebuffers(1, &frameBufferObject);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject);

    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    if (this->isSelect) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    } else {
      if (hasAlpha) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
      } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
      }
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_id, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_NONE);

    glGenRenderbuffers(1, &renderBufferObject);
    glBindRenderbuffer(GL_RENDERBUFFER, renderBufferObject);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
                              renderBufferObject);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      ERROR("Framebuffer is not complete!");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    GLuint err;
    while ((err = glGetError()) != GL_NO_ERROR) {
      ERROR("OpenGL Error: {}", err);
    }
  }

  inline void bindDraw() const {
    glViewport(0, 0, this->width, this->height);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
  }
  static void unbindDraw() {
    glDrawBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  inline void bindRead() const {
    glViewport(0, 0, this->width, this->height);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferObject);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
  }
  static void unbindRead() {
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  }

  inline void onResize(GLsizei width, GLsizei height) {

    if (width == this->width && height == this->height)
      return;

    glBindTexture(GL_TEXTURE_2D, texture_id);
    if (isSelect) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    } else {
      if (hasAlpha) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
      } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
      }
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_id, 0);

    glBindRenderbuffer(GL_RENDERBUFFER, renderBufferObject);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
                              renderBufferObject);

    this->width = width;
    this->height = height;
  }

  [[nodiscard]] inline GLuint getTextureId() const { return this->texture_id; }

private:
  GLuint frameBufferObject{};
  GLuint texture_id{};
  GLuint renderBufferObject{};

  GLsizei width = 100;
  GLsizei height = 100;

  bool isSelect;
  bool hasAlpha;

public:
  [[nodiscard]] inline GLsizei getHeight() const { return this->height; }
};

#endif // !FRAME_BUFFER_HELPER_HPP
