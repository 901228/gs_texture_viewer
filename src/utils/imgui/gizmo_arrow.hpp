#ifndef IMGUI_GIZMO_ARROW_HPP
#define IMGUI_GIZMO_ARROW_HPP
#pragma once

#include <glm/glm.hpp>

#include <ImGui/imgui.h>

namespace ImGui {

IMGUI_API bool GizmoArrow2D(const char *label, glm::vec3 &direction, const glm::vec3 &axis = {1, 0, 0},
                            const glm::vec3 &up = {0, 1, 0}, float size = 0.0f);

IMGUI_API bool GizmoArrow3D(const char *label, glm::vec3 &direction, float size = 60.0f);

} // namespace ImGui

#endif // !IMGUI_GIZMO_ARROW_HPP
