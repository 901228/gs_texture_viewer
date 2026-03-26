#include "model_panel.hpp"

#include <memory>

#include <glad/gl.h>

#include "main_window.hpp"
#include "utils/camera/camera.hpp"
#include "utils/camera/trackball_camera_three.hpp"
#include "utils/imgui/gizmo_arrow.hpp"
#include "utils/imgui/opengl.hpp"
#include "utils/imgui/sidebar.hpp"
#include "utils/mesh/geodesic_splines.hpp"
#include "utils/mesh/isosurface.hpp"
#include "utils/mesh/model.hpp"
#include "utils/texture/texture_editor.hpp"
#include "utils/utils.hpp"

ModelPanel::ModelPanel() : model(nullptr), camera(nullptr), _textureEditor(nullptr) {}

ModelPanel::~ModelPanel() { detach(); }

void ModelPanel::_attach() {

  // model = std::make_unique<Model>(Utils::Path::getAssetsPath("models/ball.obj").c_str());
  model = std::make_unique<Model>(Utils::Path::getAssetsPath("models/armadillo.obj").c_str());
  camera = std::make_unique<TrackballCameraThree>(-6.0f);
  camera->setCenter(model->center());
  _textureEditor = std::make_unique<TextureEditor>(*model, true);

  // _mc = Isosurface::extractIsosurface(*model, model->boxMin(), model->boxMax(), 128);
  // Isosurface::IsosurfaceRenderer::getInstance().upload(_mc);
}

void ModelPanel::_detach() {}

void ModelPanel::_onResize(float width, float height) {

  // projection matrix
  camera->onResize(width, height);
}

void ModelPanel::_render() {

  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *drawList;

  if (ImGui::BeginOpenGL("OpenGL", {_width, _height}, false, MainWindow::flag)) {

    float backgroundColor = 1.0f;
    static const GLfloat background[] = {backgroundColor, backgroundColor, backgroundColor, 1.0f};
    static const GLfloat one = 1.0f;

    glClearColor(backgroundColor, backgroundColor, backgroundColor, 1);
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearBufferfv(GL_COLOR, 0, background);
    glClearBufferfv(GL_DEPTH, 0, &one);

    if (_viewMode == ViewMode::Model) {
      model->render(*camera, _renderSelectedOnly, wire, _renderingMode == RenderingMode::TextureCoords,
                    _renderingMode == RenderingMode::Texture, _textureEditor->selected(),
                    _textureEditor->textureList(), _textureEditor->scale(), _textureEditor->offset(),
                    _textureEditor->theta(), _textureEditor->selectedPBR(), _lightDir, _lightIntensity);

      _textureEditor->handleBrushInput(*camera, _width, _height);
    } else if (_viewMode == ViewMode::Isosurface) {
      Isosurface::IsosurfaceRenderer::getInstance().render(*camera, _lightDir);
    }

    camera->handleInput(pos);

    drawList = ImGui::GetWindowDrawList();
  }
  ImGui::EndOpenGL();

  if (_textureEditor->isGeodesic() && GeodesicSplines::debugStruct.show) {
    GeodesicSplines::debugStruct.draw(drawList, pos, camera->projectionMatrix() * camera->viewMatrix(),
                                      _width, _height);
  }
}

void ModelPanel::_renderParameterization() {

  const glm::vec2 contentSize = {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *drawList = ImGui::GetWindowDrawList();

  // draw image
  _textureEditor->renderImage();

  // draw texture coords
  std::vector<TextureLine> lines = model->getSelectedTextureLines();
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
}

void ModelPanel::_controls() {

  if (ImGui::BeginSideBar("sidebar##model_panel_sidebar")) {

    if (ImGui::BeginSideBarItem("render##model_panel_sidebar", Model::icon)) {

      ImGui::Combo("View Mode", reinterpret_cast<int *>(&_viewMode),
                   Utils::enumToImGuiCombo<ViewMode>().c_str());

      ImGui::Checkbox("wire", &wire);
      ImGui::Checkbox("render selected only", &_renderSelectedOnly);
      ImGui::Combo("Rendering Mode", reinterpret_cast<int *>(&_renderingMode),
                   Utils::enumToImGuiCombo<RenderingMode>().c_str());

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("light##model_panel_sidebar", Light::icon)) {

      ImGui::GizmoArrow2D("##Light Direction", _lightDir);
      ImGui::SliderFloat("Light Intensity", &_lightIntensity, 0.0f, 10.0f);

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("camera##model_panel_sidebar", Camera::icon)) {

      camera->controls(model->center());

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("textures##model_panel_sidebar", TextureEditor::icon)) {

      _textureEditor->controls();

      ImGui::EndSideBarItem();
    }

    ImGui::EndSideBar();
  }
}
