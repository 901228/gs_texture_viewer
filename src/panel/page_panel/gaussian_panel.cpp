

#include "gaussian_panel.hpp"

#include "../gaussian/model/gs_model.hpp"
#include "../utils/camera/imguizmo_camera.hpp"
#include "../utils/logger.hpp"
#include "../utils/utils.hpp"
#include "imgui.h"

GaussianPanel::GaussianPanel() : _gsModel(nullptr), camera(nullptr) {}

GaussianPanel::~GaussianPanel() { detach(); }

void GaussianPanel::_attach() {
  _gsModel = std::make_unique<GaussianModel>((char *)(PROJECT_DIR "/assets/gs/armadillo.ply"), 3, 0);
  camera = std::make_unique<ImGuizmoCamera>(10.0f);
}

void GaussianPanel::_detach() {}

void GaussianPanel::_onResize(float width, float height) {

  // projection matrix
  camera->onResize(width, height);
}

void GaussianPanel::_render() {

  if (ImGui::BeginChild((name() + "_frame").c_str())) {

    ImVec2 pos = ImGui::GetCursorScreenPos();

    unsigned int textureId =
        GaussianView::getInstance().render(currMode, *camera, static_cast<int>(_width),
                                           static_cast<int>(_height), {1.0f, 1.0f, 1.0f}, *_gsModel);

    ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)textureId, ImVec2(pos.x, pos.y),
                                         ImVec2(pos.x + _width, pos.y + _height), ImVec2(0, 1), ImVec2(1, 0));

    camera->handleInput(pos);

    ImGui::EndChild();
  }
}

void GaussianPanel::_renderParameterization() {}

void GaussianPanel::_controls() {

  if (ImGui::CollapsingHeader("Gaussian Render Option")) {
    ImGui::Indent();

    if (ImGui::Combo("Render Mode", reinterpret_cast<int *>(&currMode),
                     Utils::enumToCombo<GaussianView::RenderingMode>().c_str())) {
      DEBUG("change rendering mode to {}", Utils::name(currMode));
    }

    if (currMode == GaussianView::RenderingMode::Splats) {

      _gsModel->controls();
    }

    ImGui::Unindent();
  }

  camera->controls(_gsModel->center());
}
