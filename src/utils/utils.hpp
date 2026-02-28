#ifndef UTILS_HPP
#define UTILS_HPP
#pragma once

#include <string>
#include <string_view>

#include <vector_types.h>

#include <glm/glm.hpp>

#include <imgui.h>

#include <magic_enum/magic_enum.hpp>

#include "mesh/mesh.hpp"

namespace Utils {

inline glm::vec3 toGlm(const MyMesh::Point &p) { return {p[0], p[1], p[2]}; }
inline glm::vec3 toPoint(const glm::vec3 &v) { return {v.x, v.y, v.z}; }

inline glm::vec2 toGlm(const ImVec2 &p) { return {p[0], p[1]}; }
inline ImVec2 toImVec2(const glm::vec2 &v) { return {v.x, v.y}; }

inline glm::vec2 toGlm(const MyMesh::TexCoord2D &p) { return {p[0], p[1]}; }

inline float2 toFloat2(const glm::vec2 &v) { return {v.x, v.y}; }

template <typename EnumType> inline std::string enumToCombo() {

  std::string result;
  for (const std::string_view _name : magic_enum::enum_names<EnumType>()) {
    result += _name;
    result += '\0';
  }
  return result;
}
template <typename EnumType> inline std::string_view name(EnumType value) {
  return magic_enum::enum_name(value);
}

namespace FileDialog {

std::string openImageDialog();

} // namespace FileDialog

} // namespace Utils

#endif // !UTILS_HPP
