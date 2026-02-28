#include "texture_editor.hpp"

#include <filesystem>
#include <format>

#include <imgui.h>

#include "../utils.hpp"
#include "../utils/imgui/image_selectable.hpp"
#include "../utils/imgui/tool_line.hpp"

TextureEditor::TextureEditor(const std::string &textureListPath, float scaleStep, float scaleMin,
                             float scaleMax)
    : _textureList(ImageTexture::loadTextureList(textureListPath)), _scaleStep(scaleStep),
      _scaleMin(scaleMin), _scaleMax(scaleMax) {}

TextureEditor::~TextureEditor() = default;

namespace {
std::tuple<ImVec2, ImVec2, ImVec2, ImVec2> rotateMoveTexture(const float theta, const glm::vec2 &p1,
                                                             const glm::vec2 &p2, const glm::vec2 &p3,
                                                             const glm::vec2 &p4, const glm::vec2 &center,
                                                             const glm::vec2 &offset) {
  const float rad = glm::radians(theta);
  const float cosTheta = std::cosf(rad);
  const float sinTheta = std::sinf(rad);
  const glm::mat2 rotationMatrix{cosTheta, -sinTheta, sinTheta, cosTheta};

  return std::make_tuple(Utils::toImVec2(rotationMatrix * (p1 - center) + center + offset),
                         Utils::toImVec2(rotationMatrix * (p2 - center) + center + offset),
                         Utils::toImVec2(rotationMatrix * (p3 - center) + center + offset),
                         Utils::toImVec2(rotationMatrix * (p4 - center) + center + offset));
}
} // namespace

void TextureEditor::renderImage(float repeatSize) {

  if (_selectedTexture < 0 || _selectedTexture >= _textureList.size())
    return;

  const glm::vec2 contentSize = {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *drawList = ImGui::GetWindowDrawList();

  const ImVec2 uv1{0 - repeatSize + 1, 1 + repeatSize - 1};
  const ImVec2 uv2{1 + repeatSize - 1, 1 + repeatSize - 1};
  const ImVec2 uv3{1 + repeatSize - 1, 0 - repeatSize + 1};
  const ImVec2 uv4{0 - repeatSize + 1, 0 - repeatSize + 1};

  const glm::vec2 padding = (1.0f - 1.0f / _scale) * (contentSize / 2.0f);
  const glm::vec2 center = {pos.x + contentSize.x / 2.0f, pos.y + contentSize.y / 2.0f};
  const glm::vec2 textureSize = contentSize - padding * 2.0f;

  const glm::vec2 p_min = center - (repeatSize + 0.5f - 1) * textureSize;
  const glm::vec2 p_max = center + (repeatSize + 0.5f - 1) * textureSize;
  const auto [p1, p2, p3, p4] = rotateMoveTexture(_theta, p_min, {p_max.x, p_min.y}, p_max,
                                                  {p_min.x, p_max.y}, center, _offset * contentSize);

  drawList->AddImageQuad((ImTextureID)(intptr_t)_textureList[_selectedTexture]->id(), p1, p2, p3, p4, uv1,
                         uv2, uv3, uv4);
}

void TextureEditor::renderList() {

  ImVec2 avail = ImGui::GetContentRegionAvail();
  // this value is 21, but the actual value should be 17
  avail.y -= ImGui::GetStyle().IndentSpacing;
  if (ImGui::BeginListBox("##texture list", avail)) {

    const float imageWidth = ImGui::GetContentRegionAvail().x;

    for (int i = 0; i < _textureList.size(); i++) {

      if (ImGui::ImageSelectable(std::format("##{} selectable", _textureList[i]->path()).c_str(),
                                 (ImTextureID)(intptr_t)_textureList[i]->id(), _selectedTexture == i,
                                 {imageWidth, imageWidth / _textureList[i]->aspect()},
                                 _textureList[i]->name())) {
        _selectedTexture = i;
      }
    }

    ImGui::EndListBox();
  }
}

bool TextureEditor::add(const std::string &imagePath) {

  // check image path is not empty and exists
  if (imagePath.empty() || !std::filesystem::exists(imagePath))
    return false;

  // check image path is not already in texture list
  for (const auto &i : _textureList) {
    if (i->path() == imagePath)
      return false;
  }

  _textureList.push_back(ImageTexture::create(imagePath));
  ImageTexture::saveTextureList(_textureList);

  return true;
}

void TextureEditor::handleInput() {

  const glm::vec2 contentSize = {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  ImVec2 pos = ImGui::GetCursorScreenPos();

  const ImGuiIO io = ImGui::GetIO();
  if (ImGui::IsWindowHovered()) {

    // scale
    float wheelData = io.MouseWheel;
    if (wheelData != 0) {
      _scale = std::clamp(_scale - wheelData * _scaleStep, _scaleMin, _scaleMax);
    }

    // rotate
    if (_mode != TextureEditingMode::Rotate && ImGui::IsKeyDown(ImGuiKey_R)) {
      _mode = TextureEditingMode::Rotate;
    }
    // scale
    // else if (_mode != TextureEditingMode::Scale && ImGui::IsKeyDown(ImGuiKey_S)) {
    //   _mode = TextureEditingMode::Scale;
    // }
    // move
    else if (_mode != TextureEditingMode::Move && !_isMouseLeftDown &&
             ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      _mode = TextureEditingMode::Move;
      _offsetAnchor = {_offset.x, _offset.y};
      _isMouseLeftDown = true;
      _mousePosAnchor = Utils::toGlm(ImGui::GetMousePos());
    }
  }

  // rotate
  bool isConfirm;
  if (_mode == TextureEditingMode::Rotate &&
      ImGui::ToolLineAngle("rotate texture tool", &_theta,
                           {pos.x + contentSize.x / 2, pos.y + contentSize.y / 2}, &isConfirm)) {
    _mode = TextureEditingMode::None;
  }

  // move
  if (_mode == TextureEditingMode::Move) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

    const ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
    _offset = _offsetAnchor + (Utils::toGlm(ImGui::GetMousePos()) - _mousePosAnchor) / contentSize;

    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      // cancel
      _mode = TextureEditingMode::None;
      _offset = {_offsetAnchor.x, _offsetAnchor.y};
    } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      _mode = TextureEditingMode::None;
    }
  }

  if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    _isMouseLeftDown = false;
  }
}
