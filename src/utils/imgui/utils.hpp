#ifndef IMGUI_UTILS_HPP
#define IMGUI_UTILS_HPP
#pragma once

#include <functional>
#include <string>

#include <ImGui/imgui.h>

namespace ImGui {

inline bool SliderFloat(const char *label, std::function<const float()> getter,
                        std::function<void(float)> setter, float min, float max, const char *format = "%.3f",
                        ImGuiSliderFlags flags = 0) {

  float val = getter();
  bool ret = ImGui::SliderFloat(label, &val, min, max, format, flags);
  if (ret)
    setter(val);
  return ret;
}

inline bool Checkbox(const char *label, std::function<const bool()> getter,
                     std::function<void(bool)> setter) {

  bool val = getter();
  bool ret = ImGui::Checkbox(label, &val);
  if (ret)
    setter(val);
  return ret;
}

IMGUI_API inline void SetCursorPosUnion(const ImVec2 &pos, bool isGlobal = false) {
  if (!isGlobal) {
    SetCursorPos(pos);
  } else {
    SetCursorScreenPos(pos);
  }
}

IMGUI_API inline void HCenterText(const char *text, ImVec2 p_min = {0, 0},
                                  ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false) {
  ImVec2 textPos{p_min.x + (p_max.x - p_min.x - CalcTextSize(text).x) * 0.5f, p_min.y};
  SetCursorPosUnion(textPos, isGlobal);

  Text(text);
}
IMGUI_API inline void HCenterText(const std::string &text, ImVec2 p_min = {0, 0},
                                  ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false) {
  HCenterText(text.c_str(), p_min, p_max, isGlobal);
}

IMGUI_API inline void VCenterText(const char *text, ImVec2 p_min = {0, 0},
                                  ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false) {
  ImVec2 textPos{p_min.x, p_min.y + (p_max.y - p_min.y - CalcTextSize(text).y) * 0.5f};
  SetCursorPosUnion(textPos, isGlobal);

  Text(text);
}
IMGUI_API inline void VCenterText(const std::string &text, ImVec2 p_min = {0, 0},
                                  ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false) {
  VCenterText(text.c_str(), p_min, p_max, isGlobal);
}

IMGUI_API inline void CenterText(const char *text, ImVec2 p_min = {0, 0},
                                 ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false) {
  ImVec2 textPos = p_min + (p_max - p_min - CalcTextSize(text)) * 0.5f;
  SetCursorPosUnion(textPos, isGlobal);

  Text(text);
}
IMGUI_API inline void CenterText(const std::string &text, ImVec2 p_min = {0, 0},
                                 ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false) {
  CenterText(text.c_str(), p_min, p_max, isGlobal);
}

}; // namespace ImGui

#endif // !IMGUI_UTILS_HPP
