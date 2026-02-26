#ifndef IMGUI_TOOL_LINE_H
#define IMGUI_TOOL_LINE_H
#pragma once

#include <imgui.h>

namespace ImGui {

bool ToolLineAngle(const char *label, float *angle, ImVec2 origin, bool *isConfirm, float segmentLength = 4,
                   float lineWidth = 2, ImColor lineColor = {0, 0, 0, 255});
bool ToolLineDistance(const char *label, float *distance, ImVec2 origin, bool *isConfirm,
                      float segmentLength = 4, float lineWidth = 2, ImColor lineColor = {0, 0, 0, 255});
} // namespace ImGui

#endif // !IMGUI_TOOL_LINE_H
