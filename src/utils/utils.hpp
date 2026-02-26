#ifndef UTILS_HPP
#define UTILS_HPP
#pragma once

#include <glm/glm.hpp>

#include <imgui.h>

#include "mesh/mesh.hpp"

namespace Utils {

inline glm::vec3 toGlm(const MyMesh::Point &p) { return {p[0], p[1], p[2]}; }
inline glm::vec3 toPoint(const glm::vec3 &v) { return {v.x, v.y, v.z}; }

inline glm::vec2 toGlm(const ImVec2 &p) { return {p[0], p[1]}; }
inline ImVec2 toImVec2(const glm::vec2 &v) { return {v.x, v.y}; }

inline glm::vec2 toGlm(const MyMesh::TexCoord2D &p) { return {p[0], p[1]}; }

} // namespace Utils

#endif // !UTILS_HPP
