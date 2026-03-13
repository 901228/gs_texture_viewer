#ifndef IMGUI_SIDEBAR_HPP
#define IMGUI_SIDEBAR_HPP
#pragma once

#include <ImGui/imgui.h>

class ImageTexture;

namespace ImGui {

IMGUI_API bool BeginSideBar(const char *str_id, const ImVec2 &size = {0, 0});
IMGUI_API void EndSideBar();

IMGUI_API bool BeginSideBarItem(const char *str_id, const char *icon);
IMGUI_API void EndSideBarItem();

} // namespace ImGui

#endif // !IMGUI_SIDEBAR_HPP
