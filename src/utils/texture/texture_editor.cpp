#include "texture_editor.hpp"

#include <filesystem>
#include <format>

#include <ImGui/imgui.h>

#include "../imgui/image_selectable.hpp"
#include "../imgui/tool_line.hpp"
#include "../utils.hpp"
#include "texture.hpp"

TextureEditor::TextureEditor(Model &model, bool isPBR, const std::string_view textureListPath,
                             float scaleStep, float scaleMin, float scaleMax)
    : _model(model), _isPBR(isPBR), _textureListPath(textureListPath), _scaleStep(scaleStep),
      _scaleMin(scaleMin), _scaleMax(scaleMax) {
  if (!isPBR) {
    _textureList = ImageTexture::loadTextureList(textureListPath);
  } else {
    _pbrTextureList = PBRTexture::loadTextureList(textureListPath);
  }
}

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

  if (_selectedTexture < 0 || (!_isPBR && _selectedTexture >= _textureList.size()) ||
      (_isPBR && _selectedTexture >= _pbrTextureList.size()))
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

  drawList->AddImageQuad((ImTextureID)(intptr_t)(!_isPBR
                                                     ? _textureList[_selectedTexture]->id()
                                                     : _pbrTextureList[_selectedTexture]->basecolor().id()),
                         p1, p2, p3, p4, uv1, uv2, uv3, uv4);
}

void TextureEditor::renderList() {

  ImVec2 avail = ImGui::GetContentRegionAvail();
  if (ImGui::BeginListBox("##texture list", avail)) {

    const float imageWidth = ImGui::GetContentRegionAvail().x;

    auto renderSelectable = [&](int i, const ImageTexture &texture, std::string tooltip) {
      if (ImGui::ImageSelectable(std::format("##{} selectable", texture.path()).c_str(),
                                 (ImTextureID)(intptr_t)texture.id(), _selectedTexture == i,
                                 {imageWidth, imageWidth / texture.aspect()}, tooltip)) {
        _selectedTexture = i;
        _model.updateTexId(*this);
      }
    };

    if (!_isPBR) {
      for (int i = 0; i < _textureList.size(); i++) {
        renderSelectable(i, *_textureList[i], _textureList[i]->name());
      }
    } else {
      for (int i = 0; i < _pbrTextureList.size(); i++) {
        renderSelectable(i, _pbrTextureList[i]->basecolor(), _pbrTextureList[i]->name());
      }
    }

    ImGui::EndListBox();
  }
}

void TextureEditor::controls() {

  ImGui::SeparatorText("Brush Options");
  {
    ImGui::SliderInt("Brush Size", &_brushRadius, 1, 60);
    if (ImGui::Button("Clear Selection", {ImGui::GetContentRegionAvail().x, 0})) {
      _model.clearSelect();
    }
  }
  ImGui::NewLine();

  ImGui::SeparatorText("Solving Texture Coords");
  {
    ImGui::Checkbox("Auto Solve Texture Coords", &_autoCalculate);
    ImGui::Combo("Method", reinterpret_cast<int *>(&_solvingMode),
                 Utils::enumToImGuiCombo<SolveUV::SolvingMode>().c_str());
    if (ImGui::Button("Calculate Parameterization", {ImGui::GetContentRegionAvail().x, 0})) {
      // TODO: implement angle ?
      _model.calculateParameterization(_solvingMode, 0.0f);
      _solved = true;
    }
  }
  ImGui::NewLine();

  ImGui::SeparatorText("Texture");
  {
    if (selectedPBR() != nullptr)
      selectedPBR()->controls();

    if (_autoCalculate && !_solved) {

      _model.calculateParameterization(_solvingMode, 0.0f);
      _solved = true;
    }

    if (ImGui::Button("Add Texture", {ImGui::GetContentRegionAvail().x, 0})) {

      if (!_isPBR) {
        add(Utils::File::pickImage());
      } else {
        add(Utils::File::pickFolder(), 0.0f);
      }
    }

    renderList();
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
  ImageTexture::saveTextureList(_textureList, _textureListPath);

  return true;
}

bool TextureEditor::add(const std::string &textrueDirectory, float heightScale) {

  // check image path is not empty and exists
  if (textrueDirectory.empty() || !std::filesystem::exists(textrueDirectory) ||
      !std::filesystem::is_directory(textrueDirectory)) {

    ERROR("Directory not found: {}", textrueDirectory);
    return false;
  }

  // // check image path is not already in texture list
  // for (const auto &i : _textureList) {
  //   if (i->path() == textrueDirectory)
  //     return false;
  // }

  auto textrueDirectoryPath = std::filesystem::path(textrueDirectory);
  auto checkPath = [&](const std::string &name) {
    std::optional<std::string> result;
    for (const auto &ext : Utils::File::getImageExtensions()) {
      auto path = textrueDirectoryPath / (name + ext);
      WARN("{}", path.string());
      if (std::filesystem::exists(path)) {
        result = path.string();
        break;
      }
    }
    if (!result) {
      ERROR("File not found: {}", (textrueDirectoryPath / name).string());
    }
    return result;
  };
  auto basecolorPath = checkPath("basecolor");
  auto normalPath = checkPath("normal");
  auto heightPath = checkPath("height");

  if (!basecolorPath.has_value() || !normalPath.has_value() || !heightPath.has_value())
    return false;

  _pbrTextureList.push_back(std::make_unique<PBRTexture>(
      textrueDirectory, basecolorPath.value(), normalPath.value(), heightPath.value(), heightScale));
  PBRTexture::saveTextureList(_pbrTextureList, _textureListPath);

  return true;
}

void TextureEditor::handleTextureInput() {

  if (_selectedTexture < 0 || (!_isPBR && _selectedTexture >= _textureList.size()) ||
      (_isPBR && _selectedTexture >= _pbrTextureList.size())) {

    if (ImGui::IsWindowHovered()) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_NotAllowed);
    }

    return;
  }

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

void TextureEditor::handleBrushInput(const Camera &camera, float width, float height) {

  // TODO: display brush shadow
  if (ImGui::IsWindowHovered()) {

    bool isLeftDown = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    bool isRightDown = ImGui::IsMouseDown(ImGuiMouseButton_Right);

    // handle select mesh face
    if (isLeftDown || isRightDown) {

      glm::vec2 windowPos = Utils::toGlm(ImGui::GetWindowPos());
      glm::vec2 mousePos = Utils::toGlm(ImGui::GetMousePos());
      glm::vec2 mousePosInWindow = mousePos - windowPos;
      auto [minT, selectedID, hitPos] = _model.select(camera, width, height, mousePosInWindow);

      if (selectedID >= 0 && selectedID < _model.n_faces()) {
        bool dirty = _model.selectRadius(selectedID, _brushRadius - 1, isLeftDown);
        if (dirty) {
          _model.updateTexId(*this);
        }
      }
    }

    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) || ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
      _solved = false;
    }
  }
}
