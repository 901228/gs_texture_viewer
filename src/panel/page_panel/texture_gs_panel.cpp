

#include "texture_gs_panel.hpp"

#include <imgui.h>

#include <memory>

#include "../gaussian/model/texture_gs_model.hpp"
#include "../utils/camera/trackball_camera.hpp"
#include "../utils/gl/frameBufferHelper.hpp"
#include "../utils/mesh/model.hpp"
#include "../utils/texture/texture_editor.hpp"
#include "../utils/utils.hpp"

TextureGSPanel::TextureGSPanel()
    : _textureGaussianModel(nullptr), camera(nullptr), selectingFBO(nullptr), _textureEditor(nullptr) {}

TextureGSPanel::~TextureGSPanel() { detach(); }

void TextureGSPanel::_attach() {
  _textureGaussianModel =
      std::make_unique<TextureGaussianModel>((char *)(PROJECT_DIR "/assets/gs/armadillo/geo.ply"),
                                             (char *)(PROJECT_DIR "/assets/gs/armadillo/app.ply"), 3, 0);

  selectingFBO = std::make_unique<FrameBufferHelper>(true);

  camera = std::make_unique<TrackballCamera>(-40.0f, TrackballCameraSettings());
  camera->setCenter(_textureGaussianModel->center());

  _textureEditor = std::make_unique<TextureEditor>();
}

void TextureGSPanel::_detach() {}

void TextureGSPanel::_onResize(float width, float height) {

  // projection matrix
  camera->onResize(width, height);
  selectingFBO->onResize(static_cast<GLsizei>(width), static_cast<GLsizei>(height));
}

void TextureGSPanel::_render() {

  ImVec2 pos = ImGui::GetCursorScreenPos();

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

    _textureGaussianModel->renderMesh(*camera, true, false, false, false, false, 0, {}, 1, {}, 0);

    FrameBufferHelper::unbindDraw();
  }

  if (ImGui::BeginChild(std::format("{} Splats View", name()).c_str())) {

    // render GS
    unsigned int textureId = GaussianView::getInstance().render(
        currMode, *camera, static_cast<int>(_width), static_cast<int>(_height), {1.0f, 1.0f, 1.0f},
        *_textureGaussianModel, [this](float *image_cuda) {
          auto selectedTexture = _textureEditor->selectedTexture();
          _textureGaussianModel->render(*camera, static_cast<int>(_width), static_cast<int>(_height),
                                        {1.0f, 1.0f, 1.0f}, image_cuda,
                                        selectedTexture != nullptr ? selectedTexture->cudaTextureId() : 0,
                                        {_textureEditor->scale(), Utils::toFloat2(_textureEditor->offset()),
                                         _textureEditor->theta(), _textureRenderMode});
        });

    ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)textureId, ImVec2(pos.x, pos.y),
                                         ImVec2(pos.x + _width, pos.y + _height), ImVec2(0, 1), ImVec2(1, 0));

    if (ImGui::IsWindowHovered()) {
      // handle select mesh face
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImVec2 mousePos = ImGui::GetMousePos();
        ImVec2 mousePosInWindow = ImVec2(mousePos.x - windowPos.x, mousePos.y - windowPos.y);

        // int selectedID = _textureGaussianModel->select(Utils::toGlm(mousePosInWindow));
        int selectedID = Model::getSelectedID(*selectingFBO, static_cast<int>(mousePosInWindow.x),
                                              static_cast<int>(mousePosInWindow.y));
        if (selectedID >= 0 && selectedID < _textureGaussianModel->n_faces()) {
          _textureGaussianModel->selectRadius(selectedID, brushRadius, true);
          _solved = false;
        }
      }
    }

    camera->handleInput(pos);

    ImGui::EndChild();
  }
}

void TextureGSPanel::_renderParameterization() {

  const glm::vec2 contentSize = {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *drawList = ImGui::GetWindowDrawList();

  // draw image
  _textureEditor->renderImage();

  std::vector<TextureLine> lines = _textureGaussianModel->getSelectedTextureLines();
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
  _textureEditor->handleInput();

  static const float uvPointRadius = 2;

  // draw geometry texture coords
  auto texCoords = _textureGaussianModel->getSelectedTextureCoords();
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

void TextureGSPanel::_controls() {

  if (ImGui::BeginTabBar("model panel control tab bar")) {

    if (ImGui::BeginTabItem("options")) {

      _textureGaussianModel->controls();

      ImGui::Combo("Selected Render Mode", reinterpret_cast<int *>(&_textureRenderMode),
                   Utils::enumToCombo<CudaRasterizer::RenderingMode>().c_str());

      if (ImGui::CollapsingHeader("Brush Editing Option")) {
        ImGui::Indent();

        ImGui::SliderInt("Brush Size", &brushRadius, 1, 20, "%d");
        if (ImGui::Button("Clear Selection", {ImGui::GetContentRegionAvail().x, 0})) {
          _textureGaussianModel->clearSelect();
        }

        ImGui::NewLine();
        ImGui::Combo("Method", reinterpret_cast<int *>(&_solvingMode),
                     Utils::enumToCombo<SolveUV::SolvingMode>().c_str());
        if (ImGui::Button("Calculate Parameterization", {ImGui::GetContentRegionAvail().x, 0})) {
          _textureGaussianModel->calculateParameterization(_solvingMode, 0.0f);
          _solved = true;
        }
        ImGui::Unindent();
      }

      camera->controls(_textureGaussianModel->center());

      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("textures")) {

      if (!_solved) {

        _textureGaussianModel->calculateParameterization(_solvingMode, 0.0f);
        _solved = true;
      }

      _editingTexture = true;

      if (ImGui::Button("Add Texture", {ImGui::GetContentRegionAvail().x, 0})) {

        std::string imagePath = Utils::FileDialog::openImageDialog();
        _textureEditor->add(imagePath);
      }

      _textureEditor->renderList();

      ImGui::EndTabItem();
    } else {
      _editingTexture = false;
    }

    ImGui::EndTabBar();
  }
}
