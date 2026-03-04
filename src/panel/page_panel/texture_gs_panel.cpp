

#include "texture_gs_panel.hpp"

#include <memory>

#include <ImGui/imgui.h>

#include "gaussian/model/texture_gs_model.hpp"
#include "utils/camera/trackball_camera_three.hpp"
#include "utils/mesh/model.hpp"
#include "utils/texture/texture_editor.hpp"
#include "utils/utils.hpp"

TextureGSPanel::TextureGSPanel() : _textureGaussianModel(nullptr), camera(nullptr), _textureEditor(nullptr) {}

TextureGSPanel::~TextureGSPanel() { detach(); }

void TextureGSPanel::_attach() {
  _textureGaussianModel =
      std::make_unique<TextureGaussianModel>((char *)(PROJECT_DIR "/assets/gs/armadillo/geo.ply"),
                                             (char *)(PROJECT_DIR "/assets/gs/armadillo/app.ply"), 3, 0);

  camera = std::make_unique<TrackballCameraThree>(-40.0f, TrackballCameraThreeSettings());
  camera->setCenter(_textureGaussianModel->center());

  _textureEditor = std::make_unique<TextureEditor>(*_textureGaussianModel);
}

void TextureGSPanel::_detach() {}

void TextureGSPanel::_onResize(float width, float height) {

  // projection matrix
  camera->onResize(width, height);
}

void TextureGSPanel::_render() {

  ImVec2 pos = ImGui::GetCursorScreenPos();

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
                                         _textureEditor->theta(), _textureRenderMode, _maskCullingMode});
        });

    ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)textureId, ImVec2(pos.x, pos.y),
                                         ImVec2(pos.x + _width, pos.y + _height), ImVec2(0, 1), ImVec2(1, 0));

    _textureEditor->handleBrushInput(*camera, _width, _height);

    camera->handleInput(pos);
  }
  ImGui::EndChild();
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
  _textureEditor->handleTextureInput();

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
        // drawList->AddCircleFilled({uv.first * window_width + pos.x, uv.second * window_height + pos.y},
        //                           uvPointRadius,
        //                           0xFFFF0000); // blue (ABGR)
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

      camera->controls(_textureGaussianModel->center());

      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("textures")) {

      ImGui::Combo("Selected Render Mode", reinterpret_cast<int *>(&_textureRenderMode),
                   Utils::enumToImGuiCombo<CudaRasterizer::RenderingMode>().c_str());
      ImGui::Combo("Mask Culling Mode", reinterpret_cast<int *>(&_maskCullingMode),
                   Utils::enumToImGuiCombo<CudaRasterizer::MaskCullingMode>().c_str());
      ImGui::NewLine();

      _textureEditor->controls();

      ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
  }
}
