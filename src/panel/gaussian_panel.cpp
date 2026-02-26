

#include "gaussian_panel.hpp"

#include "../gaussian/model/gs_model.hpp"
#include "../gaussian/view/gs_view.hpp"
#include "../utils/camera/imguizmo_camera.hpp"

GaussianPanel::GaussianPanel() : gaussianView(nullptr), camera(nullptr) {}

GaussianPanel::~GaussianPanel() = default;

void GaussianPanel::_init() {
  gaussianView =
      std::make_unique<GaussianView>(static_cast<int>(width), static_cast<int>(height),
                                     (char *)(PROJECT_DIR "/assets/gs/armadillo.ply"), 3, true, true, 0);
  camera = std::make_unique<ImGuizmoCamera>(10.0f);
}

void GaussianPanel::_onResize(float width, float height) {

  // projection matrix
  camera->onResize(width, height);
  gaussianView->onResize(static_cast<int>(width), static_cast<int>(height));
}

void GaussianPanel::_render() {

  if (ImGui::BeginChild((name() + "_frame").c_str())) {

    ImVec2 pos = ImGui::GetCursorScreenPos();

    gaussianView->render(*camera);

    ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)gaussianView->getTextureId(),
                                         ImVec2(pos.x, pos.y), ImVec2(pos.x + width, pos.y + height),
                                         ImVec2(0, 1), ImVec2(1, 0));

    camera->handleInput(pos);

    ImGui::EndChild();
  }
}

void GaussianPanel::_renderParameterization() {}

void GaussianPanel::_controls() {

  ImGui::SeparatorText("GS Render Option");
  {
    gaussianView->controls();
  }
  ImGui::NewLine();

  camera->controls(gaussianView->model().center());
}
