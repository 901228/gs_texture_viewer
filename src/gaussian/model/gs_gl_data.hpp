#ifndef GAUSSIAN_DATA_HPP
#define GAUSSIAN_DATA_HPP
#pragma once

#include <glad/gl.h>

/**
 * For OpenGL rendering
 */
class GaussianGLData {
public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  inline GaussianGLData(int num_gaussians, float *mean_data, float *rot_data, float *scale_data,
                        float *alpha_data, float *color_data) {

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    _num_gaussians = num_gaussians;
    glCreateBuffers(1, &meanBuffer);
    glCreateBuffers(1, &rotBuffer);
    glCreateBuffers(1, &scaleBuffer);
    glCreateBuffers(1, &alphaBuffer);
    glCreateBuffers(1, &colorBuffer);
    glNamedBufferStorage(meanBuffer, static_cast<long>(_num_gaussians * 3 * sizeof(float)), mean_data, 0);
    glNamedBufferStorage(rotBuffer, static_cast<long>(_num_gaussians * 4 * sizeof(float)), rot_data, 0);
    glNamedBufferStorage(scaleBuffer, static_cast<long>(_num_gaussians * 3 * sizeof(float)), scale_data, 0);
    glNamedBufferStorage(alphaBuffer, static_cast<long>(_num_gaussians * sizeof(float)), alpha_data, 0);
    glNamedBufferStorage(colorBuffer, static_cast<long>(_num_gaussians * sizeof(float) * 48), color_data, 0);

    glBindVertexArray(0);
  }

  inline ~GaussianGLData() {

    glBindVertexArray(vao);
    glDeleteBuffers(1, &meanBuffer);
    glDeleteBuffers(1, &rotBuffer);
    glDeleteBuffers(1, &scaleBuffer);
    glDeleteBuffers(1, &alphaBuffer);
    glDeleteBuffers(1, &colorBuffer);
    glBindVertexArray(0);

    glDeleteVertexArrays(1, &vao);
  }

  inline void render(int G) const {
    glBindVertexArray(vao);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, meanBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rotBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, scaleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, alphaBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, colorBuffer);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, G);

    glBindVertexArray(0);
  }

  inline void renderPoints(int G) const {
    glBindVertexArray(vao);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, meanBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, colorBuffer);
    glDrawArrays(GL_POINTS, 0, G);

    glBindVertexArray(0);
  }

private:
  int _num_gaussians;
  unsigned int meanBuffer;
  unsigned int rotBuffer;
  unsigned int scaleBuffer;
  unsigned int alphaBuffer;
  unsigned int colorBuffer;

  unsigned int vao;
};

#endif // !GAUSSIAN_DATA_HPP
