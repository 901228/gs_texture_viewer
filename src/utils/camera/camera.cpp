#include "camera.hpp"

Camera::Camera(CameraSettings &settings) : _settings(settings) {}

Camera::~Camera() = default;

void Camera::onResize(float width, float height) {
  if (_width == width && _height == height)
    return;

  _width = width;
  _height = height;

  _projectionMatrix = glm::perspective(fov(), aspect(), _settings.nearPlane, _settings.farPlane);
  _onResize(width, height);
}

void Camera::handleInput(const ImVec2 &pos) {
  ImGuiIO &io = ImGui::GetIO();

  // handle zoom
  if (ImGui::IsWindowHovered()) {
    float wheelDelta = io.MouseWheel;

    if (wheelDelta != 0) {
      _moveMode = MoveMode::Zoom;
      _zoom(wheelDelta);
      _moveMode = MoveMode::None;
    }
  }

  glm::vec2 localMousePos = {io.MousePos.x - pos.x, io.MousePos.y - pos.y};
  glm::vec2 ndcMousePos = _getNDCPos(localMousePos);

  // on mouse down
  if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(_settings.moveButton)) {
    if (_moveMode != MoveMode::Pan && io.KeyShift) {
      _moveMode = MoveMode::Pan;
      _onPanStart(ndcMousePos);
    } else if (_moveMode != MoveMode::Rotate || _moveMode != MoveMode::RotateUp) {
      if (!_settings.canRotateUp || _insideSphere(ndcMousePos, _settings.rotateUpRadius)) {
        _moveMode = MoveMode::Rotate;
      } else {
        _moveMode = MoveMode::RotateUp;
      }
      _onRotateStart(ndcMousePos);
    }
  }

  // on mouse move
  if (ImGui::IsMouseDragging(_settings.moveButton)) {
    if (_moveMode == MoveMode::Pan && io.KeyShift) {
      _onPanMove(ndcMousePos);
    } else if (_moveMode == MoveMode::Rotate) {
      _onRotateMove(ndcMousePos);
    } else if (_moveMode == MoveMode::RotateUp) {

      // show circle
      ImVec2 ellipseSize = {_width / 2.0f, _height / 2.0f};
      ImGui::GetForegroundDrawList()->AddEllipse(pos + ellipseSize, ellipseSize * _settings.rotateUpRadius,
                                                 0xFFFFFF00, 0.0f, 0, 8.0f);
      _onRotateUpMove(ndcMousePos);
    }
  }

  if (ImGui::IsMouseReleased(_settings.moveButton)) {
    if (_moveMode == MoveMode::Pan) {
      _onPanEnd();
    } else if (_moveMode == MoveMode::Rotate || _moveMode == MoveMode::RotateUp) {
      _onRotateEnd();
    }

    _moveMode = MoveMode::None;
  }
}

void Camera::controls(const glm::vec3 &modelCenter) {

  if (ImGui::Button("Focus on Model", {ImGui::GetContentRegionAvail().x, 0})) {
    setCenter(modelCenter);
  }

  _controls();
}

void Camera::_onPanStart(const glm::vec2 &ndcMousePos) {
  _anchorMousePos = ndcMousePos;
  _anchorCenter = center();

  glm::vec3 forward = glm::normalize(center() - eye());
  _panLeft = normalize(cross(up(), forward));
  glm::vec3 up = glm::cross(forward, _panLeft);
  if (glm::length(up) < 1e-6f)
    _panUp = glm::vec3(0, 1, 0);
  else {
    _panUp = glm::normalize(up);
  }
}
void Camera::_onPanMove(const glm::vec2 &ndcMousePos) {
  glm::vec2 mouseDelta = (ndcMousePos - _anchorMousePos) * _settings.panSpeed;
  _setCenter(_anchorCenter + _panLeft * mouseDelta.x - _panUp * mouseDelta.y);
}
void Camera::_onPanEnd() {
  _anchorMousePos = {};
  _panUp = {};
  _panLeft = {};
}
