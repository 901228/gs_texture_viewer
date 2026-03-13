#include "sidebar.hpp"

#include <format>
#include <stack>
#include <unordered_map>

#include <ImGui/imgui_internal.h>

#include <IconsFont/IconsLucide.h>

#include "utils/imgui/icon.hpp"

namespace ImGui {

static const ImVec2 iconSize = {16, 16};
static const ImVec2 iconPadding = {4, 4};
static const ImVec2 iconMargin = {2, 2};
static const ImVec2 iconFullSize = iconSize + iconPadding * 2 + iconMargin * 2;

static const float separatorWidth = 1.0f;
static const ImVec2 separatorPadding = {1, 0};

} // namespace ImGui

namespace {

struct SideBarStatus {
  int index = 0;
  bool iconCollapsed = true;
  float iconNameWidth = 0;
  ImRect sideBarRect{};
  ImRect contentRect{};
  std::unordered_map<ImGuiID, int> itemIndexStack{};

  SideBarStatus() {}
  SideBarStatus(const ImRect &sideBarRect, const ImRect &contentRect)
      : index(0), sideBarRect(sideBarRect), contentRect(contentRect) {}

  ImVec2 getSideBarDrawPos(int index) {
    return {sideBarRect.Min.x, sideBarRect.Min.y + ImGui::iconFullSize.y * index};
  }

  /**
   * @param bb is the bounding box of the icon in global space
   */
  bool drawSidebarIcon(const char *id, int pos_idx, bool selected, const char *icon, std::string tooltip) {

    ImGuiID uid = ImGui::GetID(id);
    ImVec2 pos = getSideBarDrawPos(pos_idx);
    ImRect bb{pos + ImGui::iconMargin, pos + ImGui::iconFullSize - ImGui::iconMargin};

    // register item (so ImGui knows that this item exists)
    ImGui::SetCursorScreenPos(pos);
    ImGui::ItemSize(bb.GetSize(), 0.0f);
    if (!ImGui::ItemAdd(bb, uid))
      return false;

    // handle click
    bool hovered, held;
    bool pressed = ImGui::ButtonBehavior(bb, uid, &hovered, &held);

    // draw background
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    const ImU32 bgColor = (held && hovered) ? ImGui::GetColorU32(ImGuiCol_ButtonActive)
                          : hovered         ? ImGui::GetColorU32(ImGuiCol_HeaderHovered)
                          : selected        ? ImGui::GetColorU32(ImGuiCol_Header)
                                            : 0x00FFFFFF;

    drawList->AddRectFilled(bb.Min, bb.Max, bgColor, 4.0f);

    // draw icon
    ImGui::CenterIconText(icon, bb.Min, bb.Max, true);
    if (!iconCollapsed) {

      ImVec2 textSize = ImGui::CalcTextSize(id, 0, true);
      iconNameWidth = std::max(iconNameWidth, textSize.x);
      printf("%f\n", iconNameWidth);
      ImGui::RenderText({pos.x + ImGui::iconFullSize.x, pos.y + (ImGui::iconFullSize.y - textSize.y) * 0.5f},
                        id);
    }

    // tooltip
    if (!tooltip.empty() && hovered && ImGui::BeginTooltip()) {
      const char *end = ImGui::FindRenderedTextEnd(tooltip.c_str());
      ImGui::TextUnformatted(tooltip.c_str(), end);
      ImGui::EndTooltip();
    }

    return pressed;
  };
};

} // namespace

namespace ImGui {

static std::stack<ImGuiID> SideBarIDStack;
static std::unordered_map<ImGuiID, SideBarStatus> SideBarStatusMap;

bool BeginSideBar(const char *str_id, const ImVec2 &size) {

  int beginFlag = BeginChild(str_id, size);
  if (!beginFlag) {
    return beginFlag;
  }

  const ImGuiID itemID = GetID(str_id);
  SideBarIDStack.push(itemID);

  // global pos (DrawList have to use global pos)
  ImVec2 startPos = GetCursorScreenPos();
  const ImVec2 childSize = GetContentRegionAvail();
  const ImVec2 sideBarSize = {iconFullSize.x, childSize.y};
  const ImVec2 separatorStart = {startPos.x + sideBarSize.x + separatorPadding.x, startPos.y};
  const ImVec2 separatorSize = {separatorWidth + separatorPadding.x * 2, childSize.y};
  const ImVec2 mainPadding = GetStyle().WindowPadding;

  SideBarStatus &status = SideBarStatusMap[itemID];

  status.sideBarRect = {startPos, startPos + sideBarSize};
  status.contentRect = {
      {startPos.x + sideBarSize.x + separatorSize.x + mainPadding.x, startPos.y + mainPadding.y},
      startPos + childSize - mainPadding};
  if (!status.iconCollapsed) {
    status.contentRect.Min.x += status.iconNameWidth;
  }

  // icon collapse button
  if (status.drawSidebarIcon(std::format("##{}{}", str_id, "icon collapse button").c_str(), 0, false,
                             status.iconCollapsed ? ICON_LC_PANEL_LEFT_OPEN : ICON_LC_PANEL_LEFT_CLOSE, "")) {
    status.iconCollapsed = !status.iconCollapsed;
  }

  // separator
  float separatorStartX = status.iconCollapsed ? separatorStart.x : separatorStart.x + status.iconNameWidth;
  GetWindowDrawList()->AddLine({separatorStartX, separatorStart.y},
                               {separatorStartX, separatorStart.y + separatorSize.y - separatorPadding.y},
                               0xFF000000, separatorWidth);

  return beginFlag;
}

void EndSideBar() {

  if (SideBarIDStack.empty()) {
    EndChild();
    return;
  }

  SideBarIDStack.pop();
  EndChild();
}

bool BeginSideBarItem(const char *str_id, const char *icon) {

  if (SideBarIDStack.empty()) {
    return false;
  }

  const ImGuiID sideBarID = SideBarIDStack.top();
  auto it = SideBarStatusMap.find(sideBarID);
  if (it == SideBarStatusMap.end())
    return false;

  SideBarStatus &status = it->second;

  const ImGuiID itemID = GetID(str_id);
  if (!status.itemIndexStack.contains(itemID)) {
    status.itemIndexStack[itemID] = status.itemIndexStack.size();
  }
  int idx = status.itemIndexStack[itemID];

  const bool selected = idx == status.index;

  // draw icon
  // FIXME: if icon count is large, its total height is larger than available height
  if (status.drawSidebarIcon(std::format("{}{}", str_id, "##icon").c_str(), idx + 1, selected, icon,
                             str_id)) {
    status.index = idx;
  }

  // draw content
  if (!selected) {
    return false;
  }

  SetCursorScreenPos(status.contentRect.Min);
  bool beginFlag = selected && BeginChild(str_id, status.contentRect.GetSize());
  if (!beginFlag) {
    if (selected) {
      EndChild();
    }
    return false;
  }
  return beginFlag;
}

void EndSideBarItem() { EndChild(); }

} // namespace ImGui
