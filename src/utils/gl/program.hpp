#ifndef PROGRAM_HPP
#define PROGRAM_HPP
#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include <glad/gl.h>

#include "../logger.hpp"

class Program {

public:
  inline Program(const std::string &vertexShaderPath, const std::string &fragmentShaderPath,
                 const std::string &geometryShaderPath, const std::string &tessellationControlShaderPath,
                 const std::string &tessellationEvaluationShaderPath) {

    startup(vertexShaderPath, fragmentShaderPath, geometryShaderPath, tessellationControlShaderPath,
            tessellationEvaluationShaderPath);
  }

  inline Program(const char *vertexShader_s, const char *fragmentShader_s) {

    setup_shaders(vertexShader_s, fragmentShader_s);
  }

  inline ~Program() { shutdown(); }

private:
  static std::string loadShaderFile(const std::string &path) {

    std::string absPath = std::filesystem::absolute(path).string();

    std::ifstream file;
    std::string line, file_content;
    file.open(absPath);
    if (file.is_open()) {

      do {

        std::getline(file, line);
        file_content += line + '\n';
      } while (!file.eof());
      file.close();
    } else {

      ERROR("Cannot find shader file at: {}", absPath);
    }

    return file_content;
  }

  static void printShaderLog(GLuint shader) {

    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
    if (isCompiled == GL_FALSE) {
      GLint maxLength = 0;
      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

      // The maxLength includes the NULL character
      auto *errorLog = new GLchar[maxLength];
      glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

      if (strcmp(errorLog, "") != 0)
        ERROR("shader log: {}\n", errorLog);

      delete[] errorLog;
    }
  }

public:
  GLuint renderingProgram = 0;

private:
  void setup_shaders(const char *vertex_shader_content, const char *fragment_shader_content,
                     const char *geometry_shader_content = nullptr,
                     const char *tessellation_control_shader_content = nullptr,
                     const char *tessellation_evaluation_shader_content = nullptr) {

    auto compile = [&](GLenum type, const char *src) -> GLuint {
      if (!src || strcmp(src, "") == 0)
        return 0;
      GLuint s = glCreateShader(type);
      glShaderSource(s, 1, &src, nullptr);
      glCompileShader(s);
      printShaderLog(s);
      return s;
    };

    GLuint vs = compile(GL_VERTEX_SHADER, vertex_shader_content);
    GLuint tcs = compile(GL_TESS_CONTROL_SHADER, tessellation_control_shader_content);
    GLuint tes = compile(GL_TESS_EVALUATION_SHADER, tessellation_evaluation_shader_content);
    GLuint gs = compile(GL_GEOMETRY_SHADER, geometry_shader_content);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fragment_shader_content);

    renderingProgram = glCreateProgram();
    if (vs) {
      glAttachShader(renderingProgram, vs);
    }
    if (tcs) {
      glAttachShader(renderingProgram, tcs);
    }
    if (tes) {
      glAttachShader(renderingProgram, tes);
    }
    if (gs) {
      glAttachShader(renderingProgram, gs);
    }
    if (fs) {
      glAttachShader(renderingProgram, fs);
    }
    glLinkProgram(renderingProgram);

    for (GLuint s : {vs, tcs, tes, gs, fs}) {
      if (s) {
        glDeleteShader(s);
      }
    }
    glUseProgram(0);
  }

  void startup(const std::string &vertexShaderPath, const std::string &fragmentShaderPath,
               const std::string &geometryShaderPath, const std::string &tessellationControlShaderPath,
               const std::string &tessellationEvaluationShaderPath) {

    std::string vertex_shader_string;
    std::string fragment_shader_string;
    std::string geometry_shader_string;
    std::string tessellation_control_shader_string;
    std::string tessellation_evaluation_shader_string;
    if (!vertexShaderPath.empty())
      vertex_shader_string = loadShaderFile(vertexShaderPath);
    if (!fragmentShaderPath.empty())
      fragment_shader_string = loadShaderFile(fragmentShaderPath);
    if (!geometryShaderPath.empty())
      geometry_shader_string = loadShaderFile(geometryShaderPath);
    if (!tessellationControlShaderPath.empty())
      tessellation_control_shader_string = loadShaderFile(tessellationControlShaderPath);
    if (!tessellationEvaluationShaderPath.empty())
      tessellation_evaluation_shader_string = loadShaderFile(tessellationEvaluationShaderPath);

    setup_shaders(
        vertex_shader_string.c_str(), fragment_shader_string.c_str(),
        geometry_shader_string.empty() ? nullptr : geometry_shader_string.c_str(),
        tessellation_control_shader_string.empty() ? nullptr : tessellation_control_shader_string.c_str(),
        tessellation_evaluation_shader_string.empty() ? nullptr
                                                      : tessellation_evaluation_shader_string.c_str());
  }

public:
  inline void shutdown() const { glDeleteProgram(renderingProgram); }

  inline void use() const { glUseProgram(renderingProgram); }
  static inline void unUse() { glUseProgram(0); }

private:
  inline GLint getLocation(const char *name) const { return glGetUniformLocation(renderingProgram, name); }

public:
  inline void setMat4(const char *name, const float *value) const {
    GLint location;
    if ((location = getLocation(name)) == -1)
      return;

    glUniformMatrix4fv(location, 1, GL_FALSE, value);
  }
  inline void setVec3(const char *name, const float *value) const {
    GLint location;
    if ((location = getLocation(name)) == -1)
      return;

    glUniform3fv(location, 1, value);
  }
  inline void setVec2(const char *name, const float *value) const {
    GLint location;
    if ((location = getLocation(name)) == -1)
      return;

    glUniform2fv(location, 1, value);
  }
  inline void setInt(const char *name, int value) const {
    GLint location;
    if ((location = getLocation(name)) == -1)
      return;

    glUniform1i(location, value);
  }
  inline void setFloat(const char *name, float value) const {
    GLint location;
    if ((location = getLocation(name)) == -1)
      return;

    glUniform1f(location, value);
  }
};

#endif // PROGRAM_HPP
