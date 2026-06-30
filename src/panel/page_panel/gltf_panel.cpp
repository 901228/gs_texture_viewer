#include "gltf_panel.hpp"

#include <memory>

#include <glad/gl.h>

#include <cmath>

#include <glm/gtc/type_ptr.hpp>

#include "main_window.hpp"
#include "utils/camera/camera.hpp"
#include "utils/camera/trackball_camera_three.hpp"
#include "utils/imgui/opengl.hpp"
#include "utils/imgui/sidebar.hpp"
#include "utils/mesh/gltf_model.hpp"
#include "utils/texture/texture_editor.hpp"
#include "utils/utils.hpp"

GltfPanel::GltfPanel() : model(nullptr), camera(nullptr), _textureEditor(nullptr) {}

GltfPanel::~GltfPanel() { detach(); }

std::vector<Light> GltfPanel::defaultLights() {
  // A three-point rig: a strong warm key from the front-right, a soft cool fill from
  // the front-left, and a neutral rim from behind/above to separate the silhouette.
  Light key;
  key.azimuth = 300.0f;
  key.elevation = 35.0f;
  key.color = {1.0f, 0.93f, 0.82f};
  key.intensity = 3.0f;

  Light fill;
  fill.azimuth = 230.0f;
  fill.elevation = 12.0f;
  fill.color = {0.75f, 0.85f, 1.0f};
  fill.intensity = 1.2f;

  Light rim;
  rim.azimuth = 90.0f;
  rim.elevation = 50.0f;
  rim.color = {1.0f, 1.0f, 1.0f};
  rim.intensity = 1.8f;

  return {key, fill, rim};
}

void GltfPanel::_attach() {

  model = std::make_unique<GltfModel>(Utils::Path::getAssetsPath("models/mannequin/mannequin.glb").c_str());

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
  // scale navigation speed to the model so panning/zooming a large glb isn't sluggish
  camera->setMoveSpeed(glm::max(diag * 0.3f, 0.1f));
  _textureEditor = std::make_unique<TextureEditor>(*model, true);
}

void GltfPanel::_detach() {}

std::vector<Light> GltfPanel::effectiveLights() const {
  std::vector<Light> lights = _lights;
  if (_orbitAngle != 0.0f)
    for (Light &l : lights)
      if (l.animate)
        l.azimuth = std::fmod(l.azimuth + _orbitAngle, 360.0f);
  return lights;
}

void GltfPanel::_onResize(float width, float height) { camera->onResize(width, height); }

void GltfPanel::_render() {

  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImVec2 viewOrigin{};

  // advance the light orbit animation
  if (_animateOrbit)
    _orbitAngle = std::fmod(_orbitAngle + _orbitSpeed * ImGui::GetIO().DeltaTime, 360.0f);

  const std::vector<Light> lights = effectiveLights();

  bool open = ImGui::BeginOpenGL("OpenGL", {_width, _height}, false, MainWindow::flag);
  if (open) {

    viewOrigin = ImGui::GetWindowPos(); // top-left of the 3D view (matches brush picking)

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
                  _textureEditor->theta(), _textureEditor->selectedPBR(), lights, _flipNormals);

    _textureEditor->handleBrushInput(*camera, _width, _height);

    camera->handleInput(pos);
  }
  ImGui::EndOpenGL();

  // light gizmos are drawn after EndOpenGL so they sit on top of the blitted scene image
  if (open && _showLights)
    renderLightGizmos(viewOrigin.x, viewOrigin.y, lights);
}

void GltfPanel::renderLightGizmos(float originX, float originY, const std::vector<Light> &lights) {

  const glm::vec3 center = model->center();
  float radius = 0.5f * glm::length(model->boxMax() - model->boxMin());
  if (!(radius > 1e-4f))
    radius = 1.0f;

  const glm::mat4 vp = camera->projectionMatrix() * camera->viewMatrix();
  auto project = [&](const glm::vec3 &p, ImVec2 &out) -> bool {
    glm::vec4 clip = vp * glm::vec4(p, 1.0f);
    if (clip.w <= 1e-5f) // behind the camera
      return false;
    glm::vec3 ndc = glm::vec3(clip) / clip.w;
    out = ImVec2(originX + (ndc.x * 0.5f + 0.5f) * _width, originY + (1.0f - (ndc.y * 0.5f + 0.5f)) * _height);
    return true;
  };

  // Anchor every gizmo at the projected model center and radiate the arrow outward by a
  // fixed screen distance, so lights stay on-screen regardless of model size / zoom.
  ImVec2 pCenter;
  if (!project(center, pCenter))
    return;

  const float sunRadiusPx = 110.0f; // distance from center to the sun marker
  const float innerGapPx = 16.0f;   // arrowhead stops short of the center

  // foreground draw list so the gizmos sit on top of the child window that blits the scene
  ImDrawList *dl = ImGui::GetForegroundDrawList();
  dl->PushClipRect({originX, originY}, {originX + _width, originY + _height}, true);

  for (const Light &l : lights) {
    if (!l.enabled)
      continue;

    glm::vec3 incoming = -l.direction(); // from the scene toward where the light sits
    // screen-space incoming direction, via a projected world offset from the center
    ImVec2 pProbe;
    if (!project(center + incoming * radius, pProbe))
      continue;

    ImVec2 sdir{pProbe.x - pCenter.x, pProbe.y - pCenter.y};
    float slen = std::sqrt(sdir.x * sdir.x + sdir.y * sdir.y);
    if (slen < 1e-3f)
      sdir = ImVec2(0.0f, -1.0f); // light points almost along the view axis: default to up
    else {
      sdir.x /= slen;
      sdir.y /= slen;
    }

    ImVec2 pSun{pCenter.x + sdir.x * sunRadiusPx, pCenter.y + sdir.y * sunRadiusPx};
    ImVec2 pArrowEnd{pCenter.x + sdir.x * innerGapPx, pCenter.y + sdir.y * innerGapPx};

    ImU32 col = IM_COL32(static_cast<int>(glm::clamp(l.color.r, 0.0f, 1.0f) * 255.0f),
                         static_cast<int>(glm::clamp(l.color.g, 0.0f, 1.0f) * 255.0f),
                         static_cast<int>(glm::clamp(l.color.b, 0.0f, 1.0f) * 255.0f), 255);

    // arrow pointing inward (the light's travel direction: from the sun toward the model)
    dl->AddLine(pSun, pArrowEnd, col, 2.0f);
    const float hs = 10.0f;
    ImVec2 n{-sdir.y, sdir.x};
    ImVec2 b1{pArrowEnd.x + sdir.x * hs + n.x * hs * 0.5f, pArrowEnd.y + sdir.y * hs + n.y * hs * 0.5f};
    ImVec2 b2{pArrowEnd.x + sdir.x * hs - n.x * hs * 0.5f, pArrowEnd.y + sdir.y * hs - n.y * hs * 0.5f};
    dl->AddTriangleFilled(pArrowEnd, b1, b2, col);

    // sun marker with rays
    dl->AddCircleFilled(pSun, 7.0f, col);
    dl->AddCircle(pSun, 7.0f, IM_COL32(0, 0, 0, 255), 0, 1.5f);
    for (int k = 0; k < 8; ++k) {
      float a = static_cast<float>(k) * 0.7853982f; // 45 deg
      ImVec2 r0{pSun.x + std::cos(a) * 9.0f, pSun.y + std::sin(a) * 9.0f};
      ImVec2 r1{pSun.x + std::cos(a) * 13.0f, pSun.y + std::sin(a) * 13.0f};
      dl->AddLine(r0, r1, col, 1.5f);
    }
  }

  dl->PopClipRect();
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
      ImGui::Checkbox("flip normals", &_flipNormals);
      ImGui::Combo("Rendering Mode", reinterpret_cast<int *>(&_renderingMode),
                   Utils::enumToImGuiCombo<RenderingMode>().c_str());

      ImGui::EndSideBarItem();
    }

    if (ImGui::BeginSideBarItem("light##gltf_panel_sidebar", ICON_LC_LIGHTBULB)) {

      ImGui::Checkbox("show lights in view", &_showLights);

      // orbit animation: advances the shared clock; each light opts in via its own
      // "animate" checkbox below, so you can orbit some lights while others stay fixed.
      ImGui::Checkbox("animate orbit", &_animateOrbit);
      ImGui::BeginDisabled(!_animateOrbit);
      ImGui::SliderFloat("orbit speed", &_orbitSpeed, 5.0f, 360.0f, "%.0f deg/s");
      ImGui::EndDisabled();
      ImGui::SameLine();
      if (ImGui::SmallButton("reset"))
        _orbitAngle = 0.0f; // just undo the orbit offset

      // restore every light to its frame-0 state (default rig) and clear the orbit
      if (ImGui::Button("Reset Light Positions", {ImGui::GetContentRegionAvail().x, 0})) {
        _lights = defaultLights();
        _orbitAngle = 0.0f;
      }

      ImGui::NewLine();

      ImGui::BeginDisabled(_lights.size() >= MAX_LIGHTS);
      if (ImGui::Button("Add Light", {ImGui::GetContentRegionAvail().x, 0}))
        _lights.emplace_back();
      ImGui::EndDisabled();

      int toDelete = -1;
      for (int i = 0; i < static_cast<int>(_lights.size()); ++i) {
        ImGui::PushID(i);
        Light &l = _lights[i];

        ImGui::SeparatorText(("Light " + std::to_string(i + 1)).c_str());

        ImGui::Checkbox("enabled", &l.enabled);
        ImGui::SameLine();
        ImGui::Checkbox("animate", &l.animate);
        ImGui::SameLine();
        if (ImGui::SmallButton("delete"))
          toDelete = i;

        ImGui::BeginDisabled(!l.enabled);
        ImGui::SliderFloat("azimuth", &l.azimuth, 0.0f, 360.0f, "%.0f deg");
        ImGui::SliderFloat("elevation", &l.elevation, -90.0f, 90.0f, "%.0f deg");
        ImGui::SliderFloat("intensity", &l.intensity, 0.0f, 10.0f);
        ImGui::ColorEdit3("color", glm::value_ptr(l.color));
        ImGui::EndDisabled();

        ImGui::PopID();
      }
      if (toDelete >= 0)
        _lights.erase(_lights.begin() + toDelete);

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
