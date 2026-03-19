

#include "geodesic_gs_panel.hpp"

#include <memory>

#include <ImGui/imgui.h>

#include "gaussian/model/geodesic_gs_model.hpp"
#include "utils/camera/trackball_camera_three.hpp"
#include "utils/imgui/gizmo_arrow.hpp"
#include "utils/imgui/sidebar.hpp"
#include "utils/mesh/geodesic_splines.hpp"
#include "utils/mesh/model.hpp"
#include "utils/texture/texture_editor.hpp"
#include "utils/utils.hpp"

GeodesicTextureGSPanel::GeodesicTextureGSPanel()
    : _model(nullptr), camera(nullptr), _textureEditor(nullptr) {}

GeodesicTextureGSPanel::~GeodesicTextureGSPanel() { detach(); }

void GeodesicTextureGSPanel::_attach() {
  _model = std::make_unique<GeodesicGaussianModel>(
      Utils::Path::getAssetsPath("gs/armadillo/point_cloud.ply").c_str(), 3, 0);

  camera = std::make_unique<TrackballCameraThree>(-40.0f, TrackballCameraThreeSettings());
  camera->setCenter(_model->center());

  _textureEditor = std::make_unique<TextureEditor>(*_model, true);
}

void GeodesicTextureGSPanel::_detach() {}

void GeodesicTextureGSPanel::_onResize(float width, float height) {

  // projection matrix
  camera->onResize(width, height);
}

void GeodesicTextureGSPanel::_render() {

  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *drawList;

  if (ImGui::BeginChild(std::format("{} Splats View", name()).c_str())) {

    // render GS
    unsigned int textureId = GaussianView::getInstance().render(
        currMode, *camera, static_cast<int>(_width), static_cast<int>(_height), {1.0f, 1.0f, 1.0f}, *_model,
        [this](float *image_cuda) {
          auto selectedTexture = _textureEditor->selectedTexture();
          _model->resizeBuffer(static_cast<int>(_width), static_cast<int>(_height));
          _model->render(*camera, static_cast<int>(_width), static_cast<int>(_height), {1.0f, 1.0f, 1.0f},
                         image_cuda);
        });

    ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)textureId, ImVec2(pos.x, pos.y),
                                         ImVec2(pos.x + _width, pos.y + _height), ImVec2(0, 1), ImVec2(1, 0));

    _textureEditor->handleBrushInput(*camera, _width, _height);

    camera->handleInput(pos);

    drawList = ImGui::GetWindowDrawList();
  }
  ImGui::EndChild();

  if (_textureEditor->isGeodesic() && GeodesicSplines::debugStruct.show) {

    glm::mat4 proj_view_mat = camera->projectionMatrix() * camera->viewMatrix();
    GaussianModel::flipRow(proj_view_mat, 1);

    // glm::mat4 colmap_view = camera->viewMatrix();
    // GaussianModel::flipRow(colmap_view, 1);
    // GaussianModel::flipRow(colmap_view, 2);

    // glm::mat4 proj_view_mat = camera->projectionMatrix() * colmap_view;

    GeodesicSplines::debugStruct.draw(drawList, pos, proj_view_mat, _width, _height, false);
  }
}

void GeodesicTextureGSPanel::_renderParameterization() {

  const glm::vec2 contentSize = {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *drawList = ImGui::GetWindowDrawList();

  // draw image
  _textureEditor->renderImage();

  // std::vector<TextureLine> lines = _model->getSelectedTextureLines();
  // if (!lines.empty()) {

  //   // draw texture coords
  //   for (const TextureLine &line : lines) {
  //     float x0 = pos.x + line.first.first * contentSize.x;
  //     float y0 = pos.y + line.first.second * contentSize.y;
  //     float x1 = pos.x + line.second.first * contentSize.x;
  //     float y1 = pos.y + line.second.second * contentSize.y;
  //     drawList->AddLine({x0, y0}, {x1, y1}, 0xFF000000, // black
  //                       1);
  //   }
  // }

  // handle parameterization input
  _textureEditor->handleTextureInput();

  static const float uvPointRadius = 2;

  // draw geometry texture coords
  // auto texCoords = _model->getSelectedTextureCoords();
  // if (!texCoords.empty()) {
  //   const float window_width = ImGui::GetContentRegionAvail().x;
  //   const float window_height = ImGui::GetContentRegionAvail().y;

  //   for (const auto &[i, uv] : texCoords) {

  //     // geometry points
  //     if (i == 0) {
  //       // drawList->AddCircleFilled({uv.first * window_width + pos.x, uv.second * window_height + pos.y},
  //       //                           uvPointRadius,
  //       //                           0xFF0000FF); // red (ABGR)
  //     }
  //     // appearance points
  //     else if (i == 1) {
  //       // drawList->AddCircleFilled({uv.first * window_width + pos.x, uv.second * window_height + pos.y},
  //       //                           uvPointRadius,
  //       //                           0xFFFF0000); // blue (ABGR)
  //     } else {
  //       throw std::runtime_error("Unknown point type!");
  //     }
  //   }
  // }
}

void GeodesicTextureGSPanel::_controls() {

  if (ImGui::BeginSideBar("sidebar##gs_texture_panel_sidebar")) {

    if (ImGui::BeginSideBarItem("render##gs_texture_panel_sidebar", Model::icon)) {

      _model->controls();

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("light##gs_texture_panel_sidebar", Light::icon)) {

      ImGui::GizmoArrow2D("##Light Direction", _lightDir);
      ImGui::SliderFloat("Light Intensity", &_lightIntensity, 0.0f, 10.0f);

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("camera##gs_texture_panel_sidebar", Camera::icon)) {

      camera->controls(_model->center());

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("textures##gs_texture_panel_sidebar", TextureEditor::icon)) {

      _textureEditor->controls();

      ImGui::EndSideBarItem();
    }

    ImGui::EndSideBar();
  }
}
