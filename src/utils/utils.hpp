#ifndef UTILS_HPP
#define UTILS_HPP
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include <vector_types.h>

#include <glm/glm.hpp>

#include <ImGui/imgui.h>

#include <magic_enum/magic_enum.hpp>
#include <nfd.hpp>

#include "logger.hpp"
#include "mesh/mesh.hpp"

namespace Utils {

inline glm::vec3 toGlm(const MyMesh::Point &p) { return {p[0], p[1], p[2]}; }
inline glm::vec3 toPoint(const glm::vec3 &v) { return {v.x, v.y, v.z}; }

inline glm::vec2 toGlm(const ImVec2 &p) { return {p[0], p[1]}; }
inline ImVec2 toImVec2(const glm::vec2 &v) { return {v.x, v.y}; }

inline glm::vec2 toGlm(const MyMesh::TexCoord2D &p) { return {p[0], p[1]}; }

inline float2 toFloat2(const glm::vec2 &v) { return {v.x, v.y}; }
inline float3 toFloat3(const glm::vec3 &v) { return {v.x, v.y, v.z}; }

inline glm::vec3 barycentric(glm::vec3 bary, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2) {
  return bary.x * p0 + bary.y * p1 + bary.z * p2;
}
inline glm::vec2 barycentric(glm::vec3 bary, glm::vec2 p0, glm::vec2 p1, glm::vec2 p2) {
  return bary.x * p0 + bary.y * p1 + bary.z * p2;
}
inline float barycentric(glm::vec3 bary, float p0, float p1, float p2) {
  return bary.x * p0 + bary.y * p1 + bary.z * p2;
}

template <typename EnumType> inline std::string enumToImGuiCombo() {

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

inline glm::vec3 center(glm::vec3 boxmin, glm::vec3 boxmax) { return (boxmin + boxmax) * 0.5f; }

namespace File {

namespace detail {
inline const std::array ImageExtensions = {".png", ".jpg", ".jpeg"};
}

inline std::string pickImage() {

  nfdu8filteritem_t filters[1] = {{"Image", "jpg,JPG,jpeg,JPEG,png,PNG"}};

  NFD::UniquePath outPath;
  nfdresult_t result = NFD::OpenDialog(outPath, filters, 1);
  if (result == NFD_OKAY) {
    return outPath.get();
  } else if (result == NFD_CANCEL) {
    // cancel
  } else {
    ERROR("Error: {}", NFD::GetError());
  }

  return "";
}

inline std::string pickFolder() {

  NFD::UniquePath outPath;
  nfdresult_t result = NFD::PickFolder(outPath);
  if (result == NFD_OKAY) {
    return outPath.get();
  } else if (result == NFD_CANCEL) {
    // cancel
  } else {
    ERROR("Error: {}", NFD_GetError());
  }

  return "";
}

inline std::string filename(std::string path) { return std::filesystem::path(path).filename().string(); }
inline std::string stem(std::string path) { return std::filesystem::path(path).stem().string(); }

inline const decltype(detail::ImageExtensions) &getImageExtensions() { return detail::ImageExtensions; }

} // namespace File

namespace Path {

#ifndef PROJECT_DIR
#define PROJECT_DIR "."
#endif

namespace detail {
inline const std::filesystem::path AssetsDirectory = PROJECT_DIR "/assets/";
inline const std::filesystem::path ShaderDirectory = PROJECT_DIR "/shaders/";
} // namespace detail

inline const std::string getAssetsPath(const std::string &assetsName = "") {
  return (detail::AssetsDirectory / assetsName).string();
}
inline const std::string getShaderPath(const std::string &shaderName = "") {
  return (detail::ShaderDirectory / shaderName).string();
}

} // namespace Path

} // namespace Utils

#endif // !UTILS_HPP
