#include "imguizmo_camera.hpp"

#include <ImGuizmo.h>

ImGuizmoCamera::ImGuizmoCamera(float cameraDistance, ImGuizmoCameraSettings settings)
    : Camera(settings), _cameraDistance(std::abs(cameraDistance)) {

  // view matrix
  _setViewMatrix({0, 0, cameraDistance}, {0, 0, 0}, {0, 1, 0});
}

void ImGuizmoCamera::_onResize(float width, float height) {

  // projection matrix
  memcpy(_cameraProjection, glm::value_ptr(_projectionMatrix), sizeof(float) * 16);
}
void ImGuizmoCamera::_zoom(float wheelDelta) {

  // update distance
  float newDistance =
      glm::clamp(_cameraDistance - wheelDelta, _settings.cameraDistanceMin, _settings.cameraDistanceMax);

  // calculate new eye (keep direction, only change distance)
  glm::vec3 newEye = center() - forward() * newDistance;

  // build new view matrix
  _setViewMatrix(newEye, center(), up());

  _cameraDistance = newDistance;
}
void ImGuizmoCamera::_handleInput(ImVec2 pos) {

  ImGui::SetCursorPos({ImGui::GetContentRegionAvail().x - gizmoSize.x, 0});

  if (ImGui::BeginChild("imguizmo camera", gizmoSize)) {

    ImVec2 childPos = ImGui::GetCursorScreenPos();

    // ViewGizmo
    ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());

    auto modelMatrix = glm::identity<glm::mat4>();
    ImGuizmo::ViewManipulate(_cameraView, _cameraProjection, ImGuizmo::OPERATION::ROTATE, ImGuizmo::WORLD,
                             glm::value_ptr(modelMatrix), _cameraDistance, childPos, gizmoSize, 0x10101010);
    _viewMatrix = glm::make_mat4(_cameraView);

    ImGui::EndChild();
  }

  ImGui::SetCursorScreenPos(pos);
}
void ImGuizmoCamera::_setCenter(glm::vec3 newCenter) {
  glm::vec3 currentCenter = center();
  glm::vec3 direction = newCenter - currentCenter;

  // calculate new eye (keep direction, only change distance)
  glm::vec3 newEye = eye() + direction;

  // build new view matrix
  _setViewMatrix(newEye, newCenter, up());
}
void ImGuizmoCamera::_controls() {}

void ImGuizmoCamera::_setViewMatrix(glm::vec3 eye, glm::vec3 center, glm::vec3 up) {

  Camera::setViewMatrix(eye, center, up);
  memcpy(_cameraView, glm::value_ptr(_viewMatrix), sizeof(float) * 16);
}

glm::vec3 ImGuizmoCamera::eye() const {
  glm::mat3 rot(_viewMatrix);
  glm::vec3 trans(_viewMatrix[3]);
  glm::vec3 eye = -glm::transpose(rot) * trans;

  return eye;
}
glm::vec3 ImGuizmoCamera::center() const { return eye() + forward() * _cameraDistance; }
glm::vec3 ImGuizmoCamera::up() const {
  glm::vec3 up = glm::vec3(_viewMatrix[0][1], _viewMatrix[1][1], _viewMatrix[2][1]);
  return glm::normalize(up);
}
glm::vec3 ImGuizmoCamera::forward() const {
  glm::vec3 forward = -glm::vec3(_viewMatrix[0][2], _viewMatrix[1][2], _viewMatrix[2][2]);
  return glm::normalize(forward);
}
