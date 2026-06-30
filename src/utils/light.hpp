#ifndef LIGHT_HPP
#define LIGHT_HPP
#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <glm/glm.hpp>

// Maximum number of directional lights uploaded to the shader (must match the
// lights[] array size declared in the GLSL fragment shader).
constexpr int MAX_LIGHTS = 8;

// A single directional light. The direction is stored as spherical angles so the UI
// can expose simple azimuth/elevation sliders; direction() returns the world-space
// direction the light travels (i.e. from the light towards the scene).
struct Light {
  float azimuth = 0.0f;        // degrees, rotation around the world up (Y) axis
  float elevation = 45.0f;     // degrees, angle above the horizon
  glm::vec3 color{1.0f};       // linear RGB
  float intensity = 1.0f;      //
  bool enabled = true;         //

  [[nodiscard]] glm::vec3 direction() const {
    float a = glm::radians(azimuth);
    float e = glm::radians(elevation);
    // position of the light on a unit sphere; it shines toward the origin
    glm::vec3 pos{std::cos(e) * std::cos(a), std::sin(e), std::cos(e) * std::sin(a)};
    return -glm::normalize(pos);
  }
};

#endif // !LIGHT_HPP
