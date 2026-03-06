#include "gizmo_arrow.hpp"

#include <stdexcept>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "utils/utils.hpp"

namespace ImGui {

bool GizmoArrow2D(const char *label, glm::vec3 &direction, const glm::vec3 &axis, const glm::vec3 &up,
                  float size) {

  ImDrawList *drawList = ImGui::GetWindowDrawList();
  ImVec2 canvasPos = ImGui::GetCursorScreenPos();

  // set button size
  ImVec2 buttonSize = {size, size};
  if (size <= 0) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    size = std::min(avail.x, avail.y);
    buttonSize = {size, size};
  }
  ImGui::InvisibleButton(label, buttonSize);
  bool isActive = ImGui::IsItemActive();
  bool changed = false;

  // normalize direction
  glm::vec3 axisNormalized = glm::normalize(axis);
  glm::vec3 ortho = glm::cross(direction, axisNormalized);
  if (glm::length(ortho) < 1e-6f) // prevent error caused by two parallel vectors
    throw std::runtime_error("GizmoArrow2D: direction and axis are parallel!");
  ortho = glm::normalize(ortho);
  direction = glm::normalize(glm::cross(axisNormalized, ortho));

  // handle input
  if (isActive && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {

    ImVec2 currP = ImGui::GetIO().MousePos - canvasPos;
    ImVec2 prevP = currP - ImGui::GetIO().MouseDelta;

    auto toNDC = [=](const ImVec2 &p) {
      return glm::vec2((p.x / size) * 2.0f - 1.0f,
                       1.0f - (p.y / size) * 2.0f // y should be flipped
      );
    };

    glm::vec2 _prevP = toNDC(prevP);
    glm::vec2 _currP = toNDC(currP);
    float angle = glm::length(_currP - _prevP);
    if (angle != 0) {

      if (_prevP.x * _currP.y - _prevP.y * _currP.x > 0)
        angle = -angle;
      glm::quat q = glm::angleAxis(angle, axisNormalized);
      direction = q * direction;
      changed = true;
    }
  }

  // draw background
  float radius = size * 0.5f * 0.8f;
  ImVec2 center = canvasPos + buttonSize * 0.5f;
  drawList->AddRectFilled(canvasPos, canvasPos + buttonSize, 0xFFCCCCCC);
  drawList->AddCircle(center, radius, 0xFF000000);
  drawList->AddLine(center + ImVec2(radius, 0), center - ImVec2(radius, 0), 0xFF000000);
  drawList->AddLine(center + ImVec2(0, radius), center - ImVec2(0, radius), 0xFF000000);

  // draw arrow
  {

    // project direction
    glm::vec3 right = glm::cross(axisNormalized, up);
    if (glm::length(right) < 1e-6f) // prevent error caused by two parallel vectors
      throw std::runtime_error("GizmoArrow2D: direction and axis are parallel!");
    right = glm::normalize(right);
    glm::vec3 upOrtho = glm::normalize(glm::cross(right, axisNormalized));

    // project onto right-up plane (remove forward component)
    glm::vec3 projected = direction - glm::dot(direction, axisNormalized) * axisNormalized;

    // decompose into right/up components
    ImVec2 projectedDirection{glm::dot(projected, right), -glm::dot(projected, upOrtho)};
    drawList->AddLine(center, center + projectedDirection * radius, 0xFF00FFFF);
  }

  return changed;
}

bool GizmoArrow3D(const char *label, glm::vec3 &direction, float size) {

  ImDrawList *dl = ImGui::GetWindowDrawList();
  ImVec2 canvasPos = ImGui::GetCursorScreenPos();
  float radius = size * 0.5f;
  ImVec2 center(canvasPos.x + radius, canvasPos.y + radius);

  // 佔位，讓 ImGui layout 知道這個 widget 的大小
  ImGui::InvisibleButton(label, ImVec2(size, size));
  bool isHovered = ImGui::IsItemHovered();
  bool isActive = ImGui::IsItemActive();
  bool changed = false;

  // 拖曳處理
  if (isActive && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    ImVec2 delta = ImGui::GetIO().MouseDelta;

    // 水平拖曳 → 繞 Y 軸旋轉
    // 垂直拖曳 → 繞 X 軸旋轉
    float sensitivity = 0.01f;
    float yaw = -delta.x * sensitivity;
    float pitch = -delta.y * sensitivity;

    glm::mat4 rotY = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0, 1, 0));
    glm::mat4 rotX = glm::rotate(glm::mat4(1.0f), pitch, glm::vec3(1, 0, 0));
    direction = glm::vec3(rotY * rotX * glm::vec4(direction, 0.0f));
    direction = glm::normalize(direction);
    changed = true;
  }

  // 背景圓
  ImU32 bgColor = isActive    ? IM_COL32(60, 60, 80, 255)
                  : isHovered ? IM_COL32(50, 50, 70, 255)
                              : IM_COL32(40, 40, 55, 255);
  dl->AddCircleFilled(center, radius, bgColor);
  dl->AddCircle(center, radius, IM_COL32(120, 120, 150, 255), 32, 1.5f);

  // 3D → 2D 投影（簡單正交投影）
  // direction 是光的方向（從光源指向場景），arrow 畫反方向（指向光源）
  glm::vec3 dir = glm::normalize(-direction); // 顯示「光來的方向」

  // 用一個簡單的 view rotation 讓 Z 朝畫面外
  // X → 右, Y → 上, Z → 朝向觀察者
  auto project = [&](glm::vec3 v) -> ImVec2 {
    // 簡單投影：忽略 Z（正交），縮放到圓內
    float scale = radius * 0.85f;
    return ImVec2(center.x + v.x * scale,
                  center.y - v.y * scale); // Y 軸翻轉
  };

  // Arrow 的起點（尾部）和終點（箭頭）
  glm::vec3 tail = -dir * 0.3f;
  glm::vec3 head = dir;

  ImVec2 p0 = project(tail);
  ImVec2 p1 = project(head);

  // Z 深度決定顏色（朝向畫面外比較亮）
  float brightness = glm::clamp((dir.z + 1.0f) * 0.5f, 0.3f, 1.0f);
  ImU32 arrowColor = IM_COL32((int)(255 * brightness), (int)(200 * brightness), (int)(80 * brightness), 255);

  // 畫箭桿
  dl->AddLine(p0, p1, arrowColor, 2.0f);

  // 畫箭頭三角形
  glm::vec2 arrowDir2D = glm::normalize(glm::vec2(p1.x - p0.x, p1.y - p0.y));
  glm::vec2 perp(-arrowDir2D.y, arrowDir2D.x);
  float arrowSize = radius * 0.2f;

  ImVec2 arrowLeft(p1.x - arrowDir2D.x * arrowSize + perp.x * arrowSize * 0.5f,
                   p1.y - arrowDir2D.y * arrowSize + perp.y * arrowSize * 0.5f);
  ImVec2 arrowRight(p1.x - arrowDir2D.x * arrowSize - perp.x * arrowSize * 0.5f,
                    p1.y - arrowDir2D.y * arrowSize - perp.y * arrowSize * 0.5f);

  dl->AddTriangleFilled(p1, arrowLeft, arrowRight, arrowColor);

  // XYZ 軸參考線（淡色）
  ImU32 axisX = IM_COL32(100, 40, 40, 120);
  ImU32 axisY = IM_COL32(40, 100, 40, 120);
  ImU32 axisZ = IM_COL32(40, 40, 100, 120);
  float axisLen = radius * 0.6f;
  dl->AddLine(ImVec2(center.x - axisLen, center.y), ImVec2(center.x + axisLen, center.y), axisX, 1.0f);
  dl->AddLine(ImVec2(center.x, center.y + axisLen), ImVec2(center.x, center.y - axisLen), axisY, 1.0f);

  // Label
  ImGui::SameLine();
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() + radius - ImGui::GetTextLineHeight() * 0.5f);
  ImGui::Text("%s", label);

  return changed;
}

} // namespace ImGui
