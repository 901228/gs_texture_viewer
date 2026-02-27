

#include "texture_gs_panel.hpp"

#include <imgui.h>

#include <nfd.h>

#include "../utils/camera/trackball_camera.hpp"
#include "../utils/gl/frameBufferHelper.hpp"
#include "../utils/imgui/image_selectable.hpp"
#include "../utils/imgui/tool_line.hpp"
#include "../utils/logger.hpp"
#include "../utils/mesh/model.hpp"
#include "../utils/utils.hpp"

TextureGSPanel::TextureGSPanel() : textureGaussian(nullptr), camera(nullptr), selectingFBO(nullptr) {}

TextureGSPanel::~TextureGSPanel() = default;

void TextureGSPanel::_init() {
  textureGaussian = std::make_unique<TextureGSView>(
      static_cast<int>(width), static_cast<int>(height), (char *)(PROJECT_DIR "/assets/gs/armadillo/geo.ply"),
      (char *)(PROJECT_DIR "/assets/gs/armadillo/app.ply"), 3, true, true, 0);

  selectingFBO = std::make_unique<FrameBufferHelper>(true);

  camera = std::make_unique<TrackballCamera>(-40.0f, TrackballCameraSettings());
  camera->setCenter(textureGaussian->model().center());

  _textureList = ImageTexture::loadTextureList(textureListPath);
}

void TextureGSPanel::_onResize(float width, float height) {

  // projection matrix
  camera->onResize(width, height);
  textureGaussian->onResize(static_cast<int>(width), static_cast<int>(height));
  selectingFBO->onResize(static_cast<GLsizei>(width), static_cast<GLsizei>(height));
}

void TextureGSPanel::_handleSelectMesh() {

  if (ImGui::IsWindowHovered()) {
    // handle select mesh face
    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      ImVec2 windowPos = ImGui::GetWindowPos();
      ImVec2 mousePos = ImGui::GetMousePos();
      ImVec2 mousePosInWindow = ImVec2(mousePos.x - windowPos.x, mousePos.y - windowPos.y);

      int selectedID = Model::getSelectedID(*selectingFBO, static_cast<int>(mousePosInWindow.x),
                                            static_cast<int>(mousePosInWindow.y));
      if (selectedID >= 0 && selectedID < textureGaussian->mesh().n_faces()) {
        textureGaussian->mesh().selectRadius(selectedID, brushRadius, true);
        _solved = false;
      }
    }
  }
}

void TextureGSPanel::_render() {

  ImVec2 pos = ImGui::GetCursorScreenPos();
  bool isSplatsMode = (textureGaussian->mode() == TextureGSView::RenderingMode::Splats);

  // for selecting FBO
  {
    selectingFBO->bindDraw();

    float backgroundColor = 1.0f;
    static const GLfloat background[] = {backgroundColor, backgroundColor, backgroundColor, 1.0f};
    static const GLfloat one = 1.0f;

    glClearColor(backgroundColor, backgroundColor, backgroundColor, 1);
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearBufferfv(GL_COLOR, 0, background);
    glClearBufferfv(GL_DEPTH, 0, &one);

    textureGaussian->mesh().render(*camera, true, false, false, false, false, 0, {}, 1, {}, 0);

    FrameBufferHelper::unbindDraw();
  }

  if (textureGaussian->mode() == TextureGSView::RenderingMode::Mesh) {

    if (ImGui::BeginChild(std::format("{} Mesh View", name()).c_str())) {

      // render Mesh
      textureGaussian->renderMesh(*camera, false, renderSelectedOnly, false, false, true,
                                  _editingTexture ? _selectedTexture : -1, _textureList, _editingTextureScale,
                                  _editingTextureOffset, _editingTextureTheta);

      ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)textureGaussian->getTextureId(),
                                           ImVec2(pos.x, pos.y), ImVec2(pos.x + width, pos.y + height),
                                           ImVec2(0, 1), ImVec2(1, 0));

      _handleSelectMesh();

      camera->handleInput(pos);

      ImGui::EndChild();
    }
  } else if (textureGaussian->mode() == TextureGSView::RenderingMode::Splats) {

    if (ImGui::BeginChild(std::format("{} Splats View", name()).c_str())) {

      // render GS
      textureGaussian->renderGS(*camera,
                                _selectedTexture >= 0 && _selectedTexture < _textureList.size()
                                    ? _textureList[_selectedTexture]->cudaTextureId()
                                    : 0,
                                _editingTextureScale, _editingTextureOffset, _editingTextureTheta);

      ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)textureGaussian->getTextureId(),
                                           ImVec2(pos.x, pos.y), ImVec2(pos.x + width, pos.y + height),
                                           ImVec2(0, 1), ImVec2(1, 0));

      _handleSelectMesh();

      camera->handleInput(pos);

      ImGui::EndChild();
    }
  } else {
    throw std::runtime_error("Unknown rendering mode!");
  }
}

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

void TextureGSPanel::_renderParameterization() {

  const glm::vec2 contentSize = {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *drawList = ImGui::GetWindowDrawList();

  // draw image
  if (_selectedTexture >= 0 && _selectedTexture < _textureList.size()) {

    static const int repeatSize = 2;
    static const ImVec2 uv1{0 - repeatSize + 1, 1 + repeatSize - 1};
    static const ImVec2 uv2{1 + repeatSize - 1, 1 + repeatSize - 1};
    static const ImVec2 uv3{1 + repeatSize - 1, 0 - repeatSize + 1};
    static const ImVec2 uv4{0 - repeatSize + 1, 0 - repeatSize + 1};

    const glm::vec2 padding = (1.0f - 1.0f / _editingTextureScale) * (contentSize / 2.0f);
    const glm::vec2 center = {pos.x + contentSize.x / 2.0f, pos.y + contentSize.y / 2.0f};
    const glm::vec2 textureSize = contentSize - padding * 2.0f;

    const glm::vec2 p_min = center - (repeatSize + 0.5f - 1) * textureSize;
    const glm::vec2 p_max = center + (repeatSize + 0.5f - 1) * textureSize;
    const auto [p1, p2, p3, p4] =
        rotateMoveTexture(_editingTextureTheta, p_min, {p_max.x, p_min.y}, p_max, {p_min.x, p_max.y}, center,
                          _editingTextureOffset * contentSize);

    drawList->AddImageQuad((ImTextureID)(intptr_t)_textureList[_selectedTexture]->id(), p1, p2, p3, p4, uv1,
                           uv2, uv3, uv4);
  }

  std::vector<TextureLine> lines = textureGaussian->mesh().getSelectedTextureLines();
  if (!lines.empty()) {

    // draw texture coords
    for (const TextureLine &line : lines) {
      float x0 = pos.x + line.first.first * contentSize.x;
      float y0 = pos.y + line.first.second * contentSize.y;
      float x1 = pos.x + line.second.first * contentSize.x;
      float y1 = pos.y + line.second.second * contentSize.y;
      drawList->AddLine({x0, y0}, {x1, y1}, 0xFF000000, // black
                        1);
    }
  }

  // handle parameterization input
  {
    const ImGuiIO io = ImGui::GetIO();
    if (ImGui::IsWindowHovered()) {

      // scale
      float wheelData = io.MouseWheel;
      if (wheelData != 0) {
        _editingTextureScale = std::clamp(_editingTextureScale - wheelData * _editingTextureScaleStep,
                                          _editingTextureScaleMin, _editingTextureScaleMax);
      }

      // rotate
      if (_textureEditingMode != TextureEditingMode::Rotate && ImGui::IsKeyDown(ImGuiKey_R)) {
        _textureEditingMode = TextureEditingMode::Rotate;
      }
      // scale
      // else if (_textureEditingMode != TextureEditingMode::Scale && ImGui::IsKeyDown(ImGuiKey_S)) {
      //   _textureEditingMode = TextureEditingMode::Scale;
      // }
      // move
      else if (_textureEditingMode != TextureEditingMode::Move && !_isMouseLeftDown &&
               ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        _textureEditingMode = TextureEditingMode::Move;
        _editingTextureOffsetAnchor = {_editingTextureOffset.x, _editingTextureOffset.y};
        _isMouseLeftDown = true;
        _mousePosAnchor = Utils::toGlm(ImGui::GetMousePos());
      }
    }

    // rotate
    bool isConfirm;
    if (_textureEditingMode == TextureEditingMode::Rotate &&
        ImGui::ToolLineAngle("rotate texture tool", &_editingTextureTheta,
                             {pos.x + contentSize.x / 2, pos.y + contentSize.y / 2}, &isConfirm)) {
      _textureEditingMode = TextureEditingMode::None;
    }

    // move
    if (_textureEditingMode == TextureEditingMode::Move) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

      const ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
      _editingTextureOffset =
          _editingTextureOffsetAnchor + (Utils::toGlm(ImGui::GetMousePos()) - _mousePosAnchor) / contentSize;

      if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        // cancel
        _textureEditingMode = TextureEditingMode::None;
        _editingTextureOffset = {_editingTextureOffsetAnchor.x, _editingTextureOffsetAnchor.y};
      } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        _textureEditingMode = TextureEditingMode::None;
      }
    }

    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      _isMouseLeftDown = false;
    }
  }

  if (textureGaussian->mode() == TextureGSView::RenderingMode::Splats) {

    static const float uvPointRadius = 2;

    // draw geometry texture coords
    auto texCoords = textureGaussian->mesh().getSelectedTextureCoords();
    if (!texCoords.empty()) {
      const float window_width = ImGui::GetContentRegionAvail().x;
      const float window_height = ImGui::GetContentRegionAvail().y;

      for (const auto &[i, uv] : texCoords) {

        // geometry points
        if (i == 0) {
          drawList->AddCircleFilled({uv.first * window_width + pos.x, uv.second * window_height + pos.y},
                                    uvPointRadius,
                                    0xFF0000FF); // red (ABGR)
        }
        // appearance points
        else if (i == 1) {
          drawList->AddCircleFilled({uv.first * window_width + pos.x, uv.second * window_height + pos.y},
                                    uvPointRadius,
                                    0xFFFF0000); // blue (ABGR)
        } else {
          throw std::runtime_error("Unknown point type!");
        }
      }
    }
  }
}

namespace {
std::string openImageDialog() {

  std::string filepath;

  nfdu8filteritem_t filters[1] = {{"Image", "jpg,JPG,jpeg,JPEG,png,PNG"}};
  nfdopendialogu8args_t args = {nullptr};
  args.filterList = filters;
  args.filterCount = 1;

  nfdu8char_t *outPath;
  nfdresult_t result = NFD_OpenDialogU8_With(&outPath, &args);
  if (result == NFD_OKAY) {
    filepath = {outPath};
    NFD_FreePathU8(outPath);
  } else if (result == NFD_CANCEL) {
    // cancel
  } else {
    ERROR("Error: {}", NFD_GetError());
  }

  return filepath;
}
} // namespace

void TextureGSPanel::_controls() {

  if (ImGui::BeginTabBar("model panel control tab bar")) {

    if (ImGui::BeginTabItem("options")) {

      textureGaussian->controls();

      ImGui::SeparatorText("Editing Option");
      {
        ImGui::SliderInt("Brush Size", &brushRadius, 1, 20, "%d");
        if (ImGui::Button("Clear Selection", {ImGui::GetContentRegionAvail().x, 0})) {
          textureGaussian->mesh().clearSelect();
        }

        ImGui::NewLine();
        ImGui::Combo("Method", reinterpret_cast<int *>(&_solvingMode),
                     Utils::enumToCombo<SolveUV::SolvingMode>().c_str());
        if (ImGui::Button("Calculate Parameterization", {ImGui::GetContentRegionAvail().x, 0})) {
          textureGaussian->mesh().calculateParameterization(_solvingMode, 0.0f);
          _solved = true;
        }
      }
      ImGui::NewLine();

      camera->controls(textureGaussian->model().center());

      ImGui::SeparatorText("Render Option");
      {
        bool isSplatsMode = (textureGaussian->mode() == TextureGSView::RenderingMode::Splats);
        ImGui::BeginDisabled(isSplatsMode);
        bool dummy = true;
        ImGui::Checkbox("Render Selected Only", isSplatsMode ? &dummy : &renderSelectedOnly);
        ImGui::EndDisabled();
      }
      ImGui::NewLine();

      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("textures")) {

      if (!_solved) {

        textureGaussian->mesh().calculateParameterization(_solvingMode, 0.0f);
        _solved = true;
      }

      _editingTexture = true;

      if (ImGui::Button("Add Texture", {ImGui::GetContentRegionAvail().x, 0})) {

        std::string imagePath = openImageDialog();
        if (!imagePath.empty()) {

          _textureList.push_back(ImageTexture::create(imagePath));
          ImageTexture::saveTextureList(_textureList);
        }
      }

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

      ImGui::EndTabItem();
    } else {
      _editingTexture = false;
    }

    ImGui::EndTabBar();
  }
}
