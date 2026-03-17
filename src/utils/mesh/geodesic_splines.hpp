#ifndef GEODESIC_SPLINES_HPP
#define GEODESIC_SPLINES_HPP
#pragma once

#include <unordered_set>

#include <ImGui/imgui.h>

#include "hit_test.hpp"
#include "mesh.hpp"

namespace GeodesicSplines {

struct Settings {
  int m = 50;      // radial curves
  int n = 40;      // steps
  float h = 0.01f; // step size
  bool useSubSteppedProject = true;
  bool enableSmoothing = true;
  const float kappa = 1e3f; // smoothing parameter
};
inline Settings settings;

struct DebugStruct {
  bool show = true;
  glm::vec3 center;
  std::vector<std::vector<glm::vec3>> Q;
  std::vector<std::vector<glm::vec3>> T; // 對應的 tangent
  std::vector<std::vector<float>> phi;   // 每個 step j 的 phi[i]
  std::vector<std::vector<float>> theta; // 每個 step j 的 theta[i]

  float h = 0.01f; // step size

  void draw(ImDrawList *drawList, ImVec2 pos, const glm::mat4 &projview, float width, float height) {

    auto [show, p] = project(center, projview, width, height);
    if (show) {
      drawList->AddCircleFilled(pos + p, 8.f, IM_COL32(0, 255, 150, 255));
    }

    // for (const auto &qm : Q) {
    //   int total = static_cast<int>(qm.size());
    //   for (int j = 0; j < total; ++j) {
    //     auto [show, p] = project(qm[j], projview, width, height);
    //     if (!show)
    //       continue;

    //     float t = static_cast<float>(j) / std::max(total - 1, 1);

    //     // 近點（t=0）青色 → 遠點（t=1）紅色，可自行調整
    //     uint8_t r = static_cast<uint8_t>(t * 255);
    //     uint8_t g = static_cast<uint8_t>((1.f - t) * 255);
    //     uint8_t b = 150;
    //     uint8_t a = 255;
    //     ImU32 color = IM_COL32(r, g, b, a);

    //     drawList->AddCircleFilled(pos + p, 2.f, color);
    //   }
    // }

    for (int i = 0; i < (int)Q.size(); ++i) {
      auto &curve = Q[i];
      int total = (int)curve.size();
      for (int j = 0; j < total; ++j) {

        float jj = static_cast<float>(j) / std::max(total - 1, 1);
        uint8_t r = static_cast<uint8_t>(jj * 255);
        uint8_t g = static_cast<uint8_t>((1.f - jj) * 255);
        uint8_t b = 150;
        uint8_t a = 255;
        ImU32 color = IM_COL32(r, g, b, a);

        // tangent 端點 = Q[i][j] + small * tangent
        glm::vec3 t = T[i][j]; // 需要 cache tangent
        glm::vec3 end3d = curve[j] + h * t;

        auto [show0, p0] = project(curve[j], projview, width, height);
        auto [show1, p1] = project(end3d, projview, width, height);
        if (show0 && show1)
          drawList->AddLine(pos + p0, pos + p1, color, 1.f);
      }
    }
  }

private:
  static std::pair<bool, ImVec2> project(const glm::vec3 p, const glm::mat4 &projview, float width,
                                         float height) {
    // World → Clip space
    glm::vec4 clip = projview * glm::vec4(p, 1.0f);

    // Perspective divide → NDC (-1 to 1)
    if (std::abs(clip.w) < 1e-5f)
      return std::pair(false, ImVec2());
    glm::vec3 ndc = glm::vec3(clip) / clip.w;

    // 背後的點不顯示
    if (ndc.z < -1.f || ndc.z > 1.f)
      return std::pair(false, ImVec2());

    // NDC → 螢幕像素
    float sx = (ndc.x * 0.5f + 0.5f) * width;
    float sy = (1.f - (ndc.y * 0.5f + 0.5f)) * height; // Y 軸翻轉

    return std::pair(true, ImVec2(sx, sy));
  };
};
inline DebugStruct debugStruct;

void Solve(const std::unordered_set<unsigned int> &selectedID, MyMesh &originMesh, const BVH::BVH &bvh,
           HitResult hitResult);

} // namespace GeodesicSplines

#endif // !GEODESIC_SPLINES_HPP
