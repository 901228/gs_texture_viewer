#include "sidebar.hpp"

#include <stack>
#include <stdexcept>
#include <unordered_map>

#include <ImGui/imgui_internal.h>

#include "utils/utils.hpp"

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
  static bool drawSidebarIcon(const char *id, int index, bool selected, const char *icon, std::string tooltip,
                              ImVec2 pos) {
    ImGuiID uid = ImGui::GetID(id);
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
    ImU32 bgColor;
    if (hovered) {
      bgColor = ImGui::GetColorU32(ImGuiCol_HeaderHovered);
    } else if (selected) {
      bgColor = ImGui::GetColorU32(ImGuiCol_Header);
    } else {
      bgColor = 0x00FFFFFF; // transparent
    }

    drawList->AddRectFilled(bb.Min, bb.Max, bgColor, 4.0f);

    // draw icon
    ImGui::CenterText(icon, true, bb.Min, bb.Max, true);

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

  SideBarStatus status;
  try {
    status = SideBarStatusMap.at(itemID);
  } catch (std::out_of_range) {
  }

  status.sideBarRect = {startPos, startPos + sideBarSize};
  status.contentRect = {
      {startPos.x + sideBarSize.x + separatorSize.x + mainPadding.x, startPos.y + mainPadding.y},
      startPos + childSize - mainPadding};
  SideBarStatusMap.insert_or_assign(itemID, status);

  // separator
  GetWindowDrawList()->AddLine(separatorStart,
                               {separatorStart.x, separatorStart.y + separatorSize.y - separatorPadding.y},
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
  SideBarStatus status;
  try {
    status = SideBarStatusMap.at(sideBarID);
  } catch (std::out_of_range) {
    return false;
  }

  const ImGuiID itemID = GetID(str_id);
  int idx;
  try {
    idx = status.itemIndexStack.at(itemID);
  } catch (std::out_of_range) {
    idx = status.itemIndexStack.size();
    status.itemIndexStack.insert_or_assign(itemID, idx);
    SideBarStatusMap.insert_or_assign(sideBarID, status);
  }

  const bool selected = idx == status.index;

  // draw icon
  // FIXME: if icon count is large, its total height is larger than available height
  {
    if (SideBarStatus::drawSidebarIcon(std::format("{}{}", str_id, "icon").c_str(), idx, selected, icon,
                                       str_id, status.getSideBarDrawPos(idx))) {
      status.index = idx;
      SideBarStatusMap.insert_or_assign(sideBarID, status);
    }
  }

  // TODO: draw sidebar (icon + name)

  // draw content
  if (!selected) {
    return false;
  }

  SetCursorScreenPos(status.contentRect.Min);
  bool beginFlag = selected && BeginChild(str_id, status.contentRect.GetSize());
  if (!beginFlag) {
    return false;
  }
  return beginFlag;
}

void EndSideBarItem() { EndChild(); }

} // namespace ImGui
