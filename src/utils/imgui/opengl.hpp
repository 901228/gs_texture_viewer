#ifndef IMGUI_OPENGL_HPP
#define IMGUI_OPENGL_HPP
#pragma once

#include <ImGui/imgui.h>

namespace ImGui {

bool BeginOpenGL(const char *str_id, const ImVec2 &size = ImVec2(0, 0), bool border = false,
                 ImGuiWindowFlags flags = 0);
void EndOpenGL();
void ClearOpenGL();

} // namespace ImGui

#endif // !IMGUI_OPENGL_HPP
