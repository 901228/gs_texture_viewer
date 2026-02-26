#include "tool_line.hpp"

#include <cmath>
#include <string>

#include <glad/gl.h>

#include <glm/trigonometric.hpp>

#include <imgui_internal.h>

#include <stb_image.h>

namespace {

float calculateAngle(const ImVec2 origin, const ImVec2 pos) {

  float angle = glm::degrees(std::atan2f(pos.y - origin.y, pos.x - origin.x));
  if (angle < 0.0f)
    angle += 360.0f;

  return angle;
}

float getLength(ImVec2 vec) { return ImSqrt(ImLengthSqr(vec)); }

class ToolLineHelper {
public:
  static ToolLineHelper &getInstance() {
    static ToolLineHelper instance;
    return instance;
  }

  ToolLineHelper(ToolLineHelper const &) = delete;
  // Setting s;
  // Setting s2(s); // x

  void operator=(ToolLineHelper const &) = delete;
  // Setting s2;
  // s2 = s; // x

private:
  ToolLineHelper() {

    int width, height;
    loadTexture(PROJECT_DIR "/assets/icons/double_arrow.png", &texture_id, &width, &height);
    float aspect = static_cast<float>(height) / static_cast<float>(width);

    radius = getLength({halfW, aspect * halfW});
    offsetAngle = glm::degrees(std::atanf(aspect));
  }
  // Setting s; // x

private:
  static bool loadTexture(const std::string &filename, GLuint *out_texture, int *out_width, int *out_height) {

    glGenTextures(1, out_texture);
    glBindTexture(GL_TEXTURE_2D, *out_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load(filename.c_str(), out_width, out_height, &nrChannels, 4);
    if (data) {

      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *out_width, *out_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
      glGenerateMipmap(GL_TEXTURE_2D);
    } else {

      fprintf(stderr, "Failed to load texture\n");
      return false;
    }

    stbi_image_free(data);
    return true;
  }

  [[nodiscard]] ImVec2 getRotatePoint(ImVec2 anchor, float angle) const {

    angle = glm::radians(angle);
    return {anchor[0] + cosf(angle) * radius, anchor[1] + sinf(angle) * radius};
  }

  void drawArrow(const ImVec2 center, const float angle) const {

    ImVec2 p1 = getRotatePoint(center, angle - 180 + offsetAngle);
    ImVec2 p2 = getRotatePoint(center, angle - offsetAngle);
    ImVec2 p3 = getRotatePoint(center, angle + offsetAngle);
    ImVec2 p4 = getRotatePoint(center, angle + 180 - offsetAngle);
    ImGui::GetForegroundDrawList()->AddImageQuad((ImTextureID)(intptr_t)texture_id, p1, p2, p3, p4);
  }

private:
  static constexpr float halfW = 12.0f / 2.0f;

  GLuint texture_id = 0;
  float radius;
  float offsetAngle;

public:
  [[nodiscard]] float drawToolLine(const ImVec2 origin, const ImVec2 pos, const float segmentLength,
                                   const ImColor lineColor, const float lineWidth,
                                   const float arrowAngle) const {

    ImVec2 lineUnit = {origin.x - pos.x, origin.y - pos.y};
    const float length = getLength(lineUnit);
    const auto counts = static_cast<size_t>(length / (segmentLength * 2));

    // draw line
    {
      lineUnit.x = (lineUnit.x / length) * segmentLength;
      lineUnit.y = (lineUnit.y / length) * segmentLength;

      ImVec2 src = {pos.x, pos.y};
      ImVec2 dst = {pos.x + lineUnit.x, pos.y + lineUnit.y};
      for (int i = 0; i < counts; i++) {

        ImGui::GetForegroundDrawList()->AddLine(src, dst, lineColor, lineWidth);
        src.x += lineUnit.x * 2;
        src.y += lineUnit.y * 2;
        dst.x += lineUnit.x * 2;
        dst.y += lineUnit.y * 2;
      }
      ImGui::GetForegroundDrawList()->AddLine(src, origin, lineColor, lineWidth);
    }

    // draw arrow
    drawArrow(pos, arrowAngle);

    return length;
  }
};

} // namespace

namespace ImGui {

inline ImPool<float> ToolLineAngleAnchorPool; // NOLINT(cert-err58-cpp)
inline ImPool<float> ToolLineAngleValuePool;  // NOLINT(cert-err58-cpp)

bool ToolLineAngle(const char *label, float *angle, ImVec2 origin, bool *isConfirm, float segmentLength,
                   float lineWidth, ImColor lineColor) {

  bool isActive = true;

  ImGui::PushID(label);
  ImGui::PushItemWidth(2000);
  ImGui::BeginGroup();
  {
    ImVec2 pos = ImGui::GetMousePos();
    ImGuiID itemID = ImGui::GetItemID();
    ImGui::SetMouseCursor(ImGuiMouseCursor_None);

    // get current angle
    float currentAngle = calculateAngle(origin, pos);

    // store anchor angle
    float *angleAnchor = ToolLineAngleAnchorPool.GetByKey(itemID);
    if (angleAnchor == nullptr) {
      angleAnchor = ToolLineAngleAnchorPool.GetOrAddByKey(itemID);
      *angleAnchor = currentAngle;
    }

    float *angleValue = ToolLineAngleValuePool.GetByKey(itemID);
    if (angleValue == nullptr) {
      angleValue = ToolLineAngleValuePool.GetOrAddByKey(itemID);
      *angleValue = *angle;
    }

    // draw tool line
    ToolLineHelper::getInstance().drawToolLine(origin, pos, segmentLength, lineColor, lineWidth,
                                               currentAngle);

    *angle = *angleValue + *angleAnchor - currentAngle;

    // handle mouse click for comfirmation
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      // cancel
      (*isConfirm) = false;
      *angle = *angleValue;
    } else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      // confirm
      (*isConfirm) = true;
    } else {
      isActive = false;
    }

    if (isActive) {

      ToolLineAngleAnchorPool.Remove(itemID, angleAnchor);
      ToolLineAngleValuePool.Remove(itemID, angleValue);
    }
  }
  ImGui::EndGroup();
  ImGui::PopItemWidth();
  ImGui::PopID();

  return isActive;
}

inline ImPool<float> ToolLineDistanceAnchorPool; // NOLINT(cert-err58-cpp)
inline ImPool<float> ToolLineDistanceValuePool;  // NOLINT(cert-err58-cpp)

bool ToolLineDistance(const char *label, float *distance, ImVec2 origin, bool *isConfirm, float segmentLength,
                      float lineWidth, ImColor lineColor) {

  bool isActive = true;

  ImGui::PushID(label);
  ImGui::PushItemWidth(2000);
  ImGui::BeginGroup();
  {
    ImVec2 pos = ImGui::GetMousePos();
    ImGuiID itemID = ImGui::GetItemID();
    ImGui::SetMouseCursor(ImGuiMouseCursor_None);

    float *distanceValue = ToolLineDistanceValuePool.GetByKey(itemID);
    if (distanceValue == nullptr) {
      distanceValue = ToolLineDistanceValuePool.GetOrAddByKey(itemID);
      *distanceValue = *distance;
    }

    // draw tool line
    float currentDistance = ToolLineHelper::getInstance().drawToolLine(
        origin, pos, segmentLength, lineColor, lineWidth, calculateAngle(origin, pos) + 90.0f);

    // store anchor angle
    float *distanceAnchor = ToolLineDistanceAnchorPool.GetByKey(itemID);
    if (distanceAnchor == nullptr) {
      distanceAnchor = ToolLineDistanceAnchorPool.GetOrAddByKey(itemID);
      *distanceAnchor = currentDistance;
    }

    *distance = *distanceValue + (*distanceAnchor - currentDistance);

    // prevent drag window
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      // cancel
      (*isConfirm) = false;
      *distance = *distanceValue;
    } else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      // confirm
      (*isConfirm) = true;
    } else {
      isActive = false;
    }

    if (isActive) {

      ToolLineDistanceAnchorPool.Remove(itemID, distanceAnchor);
      ToolLineDistanceValuePool.Remove(itemID, distanceValue);
    }

    ImGui::SetMouseCursor(ImGuiMouseCursor_None);
  }
  ImGui::EndGroup();
  ImGui::PopItemWidth();
  ImGui::PopID();

  return isActive;
}

} // namespace ImGui
