#ifndef SCREEN_QUAD_HPP
#define SCREEN_QUAD_HPP
#pragma once

#include <memory>

#include "program.hpp"

const static char *vert_src = R"(
    #version 450 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
)";

const static char *frag_src = R"(
    #version 450 core
    in vec2 TexCoord;
    out vec4 FragColor;
    uniform sampler2D screenTexture;
    uniform float alpha;
    void main() {
        FragColor = texture(screenTexture, TexCoord);
        FragColor.a *= alpha;
    }
)";

class ScreenQuad {
public:
  inline ScreenQuad() : program(std::make_unique<Program>(vert_src, frag_src)) {
    float vertices[] = {// pos          // uv
                        -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f,
                        -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  1.0f, 1.0f};

    glGenVertexArrays(1, &vertexArrayObject);
    glGenBuffers(1, &vertexBufferObject);
    glBindVertexArray(vertexArrayObject);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glBindVertexArray(0);
  }

private:
  std::unique_ptr<Program> program;
  GLuint vertexArrayObject = 0;
  GLuint vertexBufferObject = 0;

public:
  void render(GLuint texture, float alpha = 1.0f) {
    program->use();
    glBindVertexArray(vertexArrayObject);

    program->setInt("screenTexture", 0);
    program->setFloat("alpha", alpha);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    Program::unUse();
  }
};

#endif // !SCREEN_QUAD_HPP
