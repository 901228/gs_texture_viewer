#ifndef IMGUI_IMAGE_SELECTABLE_HPP
#define IMGUI_IMAGE_SELECTABLE_HPP
#pragma once

#include <string>

#include <imgui.h>
#include <imgui_internal.h>

namespace ImGui {

inline bool ImageSelectable(const char *id, ImTextureID texture, bool selected, ImVec2 size,
                            const std::string &tooltip = "") {
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  if (window->SkipItems)
    return false;

  ImGuiStyle &style = ImGui::GetStyle();
  ImGuiID uid = ImGui::GetID(id);

  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImRect bb{pos, {pos.x + size.x, pos.y + size.y}};
  ImVec2 bb_inner_min{bb.Min.x + style.FramePadding.x, bb.Min.y + style.FramePadding.y};
  ImVec2 bb_inner_max{bb.Max.x - style.FramePadding.x, bb.Max.y - style.FramePadding.y};

  // register item (so ImGui knows that this item exists)
  ImGui::ItemSize(size, 0.0f);
  if (!ImGui::ItemAdd(bb, uid))
    return false;

  // handle click
  bool hovered, held;
  bool pressed = ImGui::ButtonBehavior(bb, uid, &hovered, &held);

  // draw background
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  ImU32 bgColor;
  if (selected) {
    bgColor = ImGui::GetColorU32(ImGuiCol_Header);
  } else if (hovered) {
    bgColor = ImGui::GetColorU32(ImGuiCol_HeaderHovered);
  } else {
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    bgColor = IM_COL32(0, 0, 0, 0); // transparent
  }

  drawList->AddRectFilled(bb.Min, bb.Max, bgColor);

  // draw image
  drawList->AddImage(texture, bb_inner_min, bb_inner_max, {0, 1}, {1, 0});

  // tooltip
  if (!tooltip.empty() && hovered && ImGui::BeginTooltip()) {
    ImGui::Text(tooltip.c_str());
    ImGui::EndTooltip();
  }

  return pressed;
}

} // namespace ImGui

#endif // !IMGUI_IMAGE_SELECTABLE_HPP
