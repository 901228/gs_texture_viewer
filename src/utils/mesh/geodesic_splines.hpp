#ifndef GEODESIC_SPLINES_HPP
#define GEODESIC_SPLINES_HPP
#pragma once

#include <glm/glm.hpp>

#include <ImGui/imgui.h>

namespace MapInterpolation {
struct PeriodicSpline;
}

namespace LogarithmicMap {

struct LogMapTable {
  std::vector<glm::vec2> uvs;   // tangent space coords
  std::vector<glm::vec3> pts3d; // corresponding 3D positions

  // Grid Acceleration
  int gridRes = 32;
  float cellSize;
  glm::vec3 gridMin;
  std::vector<std::vector<int>> grid; // gridRes^3

  void buildGrid();

  void build(const std::vector<MapInterpolation::PeriodicSpline> &isolineSplines, int n, float h,
             const glm::vec3 &origin, int numSamples = 5000);

  // given a 3D point, find the corresponding UV
  glm::vec2 query(const glm::vec3 &p) const;
};

} // namespace LogarithmicMap

namespace GeodesicSplines {

class Implicit {
public:
  // f(x) = σ(x) - threshold -> signed distance to surface
  virtual const float eval(const glm::vec3 &x) = 0;

  // ∇f(x)
  virtual const glm::vec3 grad(const glm::vec3 &x) = 0;

  // π(x) -> project onto surface
  virtual const glm::vec3 project(const glm::vec3 &x) = 0;

  // n(x) = normalize(-∇f) -> interpolated normal at projected point
  virtual const glm::vec3 normal(const glm::vec3 &x) = 0;
};

struct Settings {
  int m = 50;      // radial curves
  int n = 100;     // steps
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

  void draw(ImDrawList *drawList, ImVec2 pos, const glm::mat4 &projview, float width, float height,
            bool flip_y = true) {

    auto [show, p] = project(center, projview, width, height, flip_y);
    if (show) {
      drawList->AddCircleFilled(pos + p, 4.f, IM_COL32(0, 255, 150, 255));
    }

    // for (const auto &qm : Q) {
    //   int total = static_cast<int>(qm.size());
    //   for (int j = 0; j < total; ++j) {
    //     auto [show, p] = project(qm[j], projview, width, height, flip_y);
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

        auto [show0, p0] = project(curve[j], projview, width, height, flip_y);
        auto [show1, p1] = project(end3d, projview, width, height, flip_y);
        if (show0 && show1)
          drawList->AddLine(pos + p0, pos + p1, color, 2.f);
      }
    }
  }

private:
  static std::pair<bool, ImVec2> project(const glm::vec3 p, const glm::mat4 &projview, float width,
                                         float height, bool flip_y = true) {
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
    float sy = (ndc.y * 0.5f + 0.5f) * height;
    if (flip_y)
      sy = height - sy;

    return std::pair(true, ImVec2(sx, sy));
  };
};
inline DebugStruct debugStruct;

std::pair<LogarithmicMap::LogMapTable, float> Solve(glm::vec3 center, Implicit &model);

} // namespace GeodesicSplines

#endif // !GEODESIC_SPLINES_HPP
