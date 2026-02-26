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
                 const std::string &geometryShaderPath) {

    startup(vertexShaderPath, fragmentShaderPath, geometryShaderPath);
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
                     const char *geometry_shader_content = nullptr) {

    bool isGeomExist = !(geometry_shader_content == nullptr || strcmp(geometry_shader_content, "") == 0);

    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_content, nullptr);
    glCompileShader(vertex_shader);
    printShaderLog(vertex_shader);

    GLuint geometry_shader = 0;
    if (isGeomExist) {

      geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
      glShaderSource(geometry_shader, 1, &geometry_shader_content, nullptr);
      glCompileShader(geometry_shader);
      printShaderLog(geometry_shader);
    }

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_content, nullptr);
    glCompileShader(fragment_shader);
    printShaderLog(fragment_shader);

    renderingProgram = glCreateProgram();
    glAttachShader(renderingProgram, vertex_shader);
    if (isGeomExist)
      glAttachShader(renderingProgram, geometry_shader);
    glAttachShader(renderingProgram, fragment_shader);
    glLinkProgram(renderingProgram);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    if (isGeomExist) {
      glDeleteShader(geometry_shader);
    }

    glUseProgram(0);
  }

  void startup(const std::string &vertexShaderPath, const std::string &fragmentShaderPath,
               const std::string &geometryShaderPath) {

    std::string vertex_shader_string;
    std::string fragment_shader_string;
    std::string selecting_fragment_shader_string;
    std::string geometry_shader_string;
    if (!vertexShaderPath.empty())
      vertex_shader_string = loadShaderFile(vertexShaderPath);
    if (!fragmentShaderPath.empty())
      fragment_shader_string = loadShaderFile(fragmentShaderPath);
    if (!geometryShaderPath.empty())
      geometry_shader_string = loadShaderFile(geometryShaderPath);

    setup_shaders(vertex_shader_string.c_str(), fragment_shader_string.c_str(),
                  geometry_shader_string.empty() ? nullptr : geometry_shader_string.c_str());
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
