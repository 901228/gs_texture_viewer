#include "icon.hpp"

#include <ImGui/imgui_internal.h>

#include "imgui.h"
#include "utils.hpp"

namespace ImGui {

ImFont *iconOnlyFont = nullptr;

void PushIconFont() { PushFont(iconOnlyFont); }
void PopIconFont() { PopFont(); }

void IconOnlyText(const char *icon) {
  PushIconFont();
  Text(icon);
  PopIconFont();
}

void HCenterIconText(const char *text, ImVec2 p_min, ImVec2 p_max, bool isGlobal) {
  PushIconFont();
  HCenterText(text);
  PopIconFont();
}
void HCenterIconText(const std::string &text, ImVec2 p_min, ImVec2 p_max, bool isGlobal) {
  HCenterIconText(text.c_str(), p_min, p_max, isGlobal);
}

void VCenterIconText(const char *text, ImVec2 p_min, ImVec2 p_max, bool isGlobal) {
  PushIconFont();
  VCenterText(text);
  PopIconFont();
}
void VCenterIconText(const std::string &text, ImVec2 p_min, ImVec2 p_max, bool isGlobal) {
  VCenterIconText(text.c_str(), p_min, p_max, isGlobal);
}

void CenterIconText(const char *text, const ImVec2 &p_min, const ImVec2 &p_max, bool isGlobal) {
  PushIconFont();
  CenterText(text, p_min, p_max, isGlobal);
  PopIconFont();
}
void CenterIconText(const std::string &text, const ImVec2 &p_min, const ImVec2 &p_max, bool isGlobal) {
  CenterIconText(text.c_str(), p_min, p_max, isGlobal);
}

ImVec2 GetButtonSize(const char *label, const ImVec2 &size_arg) {
  const ImVec2 label_size = CalcTextSize(label, NULL, true);
  ImGuiContext &g = *GImGui;
  const ImGuiStyle &style = g.Style;
  ImVec2 size = CalcItemSize(size_arg, label_size.x + style.FramePadding.x * 2.0f,
                             label_size.y + style.FramePadding.y * 2.0f);
  return size;
}

} // namespace ImGui
