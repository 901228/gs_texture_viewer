#include "gltf_panel.hpp"

#include <memory>

#include <glad/gl.h>

#include "main_window.hpp"
#include "utils/camera/camera.hpp"
#include "utils/camera/trackball_camera_three.hpp"
#include "utils/imgui/gizmo_arrow.hpp"
#include "utils/imgui/opengl.hpp"
#include "utils/imgui/sidebar.hpp"
#include "utils/mesh/gltf_model.hpp"
#include "utils/texture/texture_editor.hpp"
#include "utils/utils.hpp"

GltfPanel::GltfPanel() : model(nullptr), camera(nullptr), _textureEditor(nullptr) {}

GltfPanel::~GltfPanel() { detach(); }

void GltfPanel::_attach() {

  model = std::make_unique<GltfModel>(Utils::Path::getAssetsPath("mannequin/mannequin.glb").c_str());

  // Frame the camera to the model's size: glb scenes can be far larger or smaller than the
  // sample .obj meshes, so derive the view distance and near/far planes from the bounding box.
  glm::vec3 size = model->boxMax() - model->boxMin();
  float diag = glm::length(size);
  if (!(diag > 0.0f) || diag > 1e6f)
    diag = 6.0f; // fallback when the model failed to load

  TrackballCameraThreeSettings settings(/*fov*/ 30.0f, /*near*/ diag * 0.01f, /*far*/ diag * 20.0f,
                                        /*distMin*/ diag * 0.05f, /*distMax*/ diag * 5.0f);
  camera = std::make_unique<TrackballCameraThree>(-diag * 1.3f, settings);
  camera->setCenter(model->center());
  _textureEditor = std::make_unique<TextureEditor>(*model, true);
}

void GltfPanel::_detach() {}

void GltfPanel::_onResize(float width, float height) { camera->onResize(width, height); }

void GltfPanel::_render() {

  ImVec2 pos = ImGui::GetCursorScreenPos();

  if (ImGui::BeginOpenGL("OpenGL", {_width, _height}, false, MainWindow::flag)) {

    float backgroundColor = 1.0f;
    static const GLfloat background[] = {backgroundColor, backgroundColor, backgroundColor, 1.0f};
    static const GLfloat one = 1.0f;

    glClearColor(backgroundColor, backgroundColor, backgroundColor, 1);
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearBufferfv(GL_COLOR, 0, background);
    glClearBufferfv(GL_DEPTH, 0, &one);

    model->render(*camera, _renderSelectedOnly, wire, _renderingMode == RenderingMode::TextureCoords,
                  _renderingMode == RenderingMode::Texture, _textureEditor->selected(),
                  _textureEditor->textureList(), _textureEditor->scale(), _textureEditor->offset(),
                  _textureEditor->theta(), _textureEditor->selectedPBR(), _lightDir, _lightIntensity);

    _textureEditor->handleBrushInput(*camera, _width, _height);

    camera->handleInput(pos);
  }
  ImGui::EndOpenGL();
}

void GltfPanel::_renderParameterization() {

  const glm::vec2 contentSize = {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *drawList = ImGui::GetWindowDrawList();

  _textureEditor->renderImage();

  std::vector<TextureLine> lines = model->getSelectedTextureLines();
  if (!lines.empty()) {
    for (const TextureLine &line : lines) {
      float x0 = pos.x + line.first.first * contentSize.x;
      float y0 = pos.y + line.first.second * contentSize.y;
      float x1 = pos.x + line.second.first * contentSize.x;
      float y1 = pos.y + line.second.second * contentSize.y;
      drawList->AddLine({x0, y0}, {x1, y1}, 0xFF000000, 1);
    }
  }

  _textureEditor->handleTextureInput();
}

void GltfPanel::_controls() {

  if (ImGui::BeginSideBar("sidebar##gltf_panel_sidebar")) {

    if (ImGui::BeginSideBarItem("render##gltf_panel_sidebar", GltfModel::icon)) {

      ImGui::Checkbox("wire", &wire);
      ImGui::Checkbox("render selected only", &_renderSelectedOnly);
      ImGui::Combo("Rendering Mode", reinterpret_cast<int *>(&_renderingMode),
                   Utils::enumToImGuiCombo<RenderingMode>().c_str());

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("light##gltf_panel_sidebar", ICON_LC_LIGHTBULB)) {

      ImGui::GizmoArrow2D("##Light Direction", _lightDir);
      ImGui::SliderFloat("Light Intensity", &_lightIntensity, 0.0f, 10.0f);

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("camera##gltf_panel_sidebar", Camera::icon)) {

      camera->controls(model->center());

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("textures##gltf_panel_sidebar", TextureEditor::icon)) {

      _textureEditor->controls();

      ImGui::EndSideBarItem();
    }

    ImGui::EndSideBar();
  }
}
