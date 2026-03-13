#ifndef IMGUI_ICON_HPP
#define IMGUI_ICON_HPP
#pragma once

#include <string>

#include <ImGui/imgui.h>

namespace ImGui {

extern ImFont *iconOnlyFont;

IMGUI_API void IconOnlyText(const char *icon);

IMGUI_API void PushIconFont();
IMGUI_API void PopIconFont();

IMGUI_API void HCenterIconText(const char *text, ImVec2 p_min = {0, 0},
                               ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false);
IMGUI_API void HCenterIconText(const std::string &text, ImVec2 p_min = {0, 0},
                               ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false);

IMGUI_API void VCenterIconText(const char *text, ImVec2 p_min = {0, 0},
                               ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false);
IMGUI_API void VCenterIconText(const std::string &text, ImVec2 p_min = {0, 0},
                               ImVec2 p_max = GetContentRegionAvail(), bool isGlobal = false);

IMGUI_API void CenterIconText(const char *text, const ImVec2 &p_min = {0, 0},
                              const ImVec2 &p_max = GetContentRegionAvail(), bool isGlobal = false);
IMGUI_API void CenterIconText(const std::string &text, const ImVec2 &p_min = {0, 0},
                              const ImVec2 &p_max = GetContentRegionAvail(), bool isGlobal = false);

IMGUI_API ImVec2 GetButtonSize(const char *label, const ImVec2 &size_arg = {0, 0});

} // namespace ImGui

#endif // !IMGUI_ICON_HPP
