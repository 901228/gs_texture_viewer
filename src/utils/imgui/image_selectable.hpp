#ifndef IMGUI_IMAGE_SELECTABLE_HPP
#define IMGUI_IMAGE_SELECTABLE_HPP
#pragma once

#include <string>

#include <ImGui/imgui.h>

namespace ImGui {

bool ImageSelectable(const char *id, ImTextureID texture, bool selected, ImVec2 size,
                     const std::string &tooltip = "");

} // namespace ImGui

#endif // !IMGUI_IMAGE_SELECTABLE_HPP
