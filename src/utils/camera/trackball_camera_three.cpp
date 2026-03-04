#include "trackball_camera_three.hpp"

TrackballCameraThree::TrackballCameraThree(float cameraDistance, TrackballCameraThreeSettings settings)
    : Camera(settings), _eye({0, 0, cameraDistance}) {

  _updateViewMatrix();
}

void TrackballCameraThree::_onResize(float width, float height) {}

void TrackballCameraThree::_zoom(float wheelDelta) {

  glm::vec3 forward = _eye - _center;

  // update distance
  float newDistance =
      glm::clamp(glm::length(forward) - wheelDelta, _settings.cameraDistanceMin, _settings.cameraDistanceMax);

  // calculate new eye (keep direction, only change distance)
  _eye = _center + glm::normalize(forward) * newDistance;

  // build new view matrix
  _updateViewMatrix();
}

void TrackballCameraThree::_handleInput(ImVec2 pos) {

  ImVec2 mousePos = ImGui::GetMousePos();
  glm::vec2 localMousePos = {mousePos.x - pos.x, mousePos.y - pos.y};

  // on mouse down
  if (!_isRotating && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(_trackBallSettings.rotateButton)) {
    _isRotating = true;

    _currP = _getMouseOnCircle(localMousePos);
    _prevP = _currP;
  }

  // on mouse move
  if (_isRotating && ImGui::IsMouseDragging(_trackBallSettings.rotateButton)) {

    _prevP = _currP;
    _currP = _getMouseOnCircle(localMousePos);

    glm::vec2 _cursorDirection = _currP - _prevP;
    float angle = glm::length(_cursorDirection);

    if (angle != 0) {

      // forward is from eye to center, so this is backward
      glm::vec3 backward = glm::normalize(_eye - _center);
      glm::vec3 right = glm::normalize(glm::cross(_up, backward));

      glm::vec3 up = _up * _cursorDirection.y;
      right *= _cursorDirection.x;
      glm::vec3 moveDirection = up + right;

      glm::vec3 axis = glm::cross(moveDirection, backward);
      if (glm::length(axis) < 1e-6f) // prevent error caused by two parallel vectors
        return;
      axis = glm::normalize(axis);

      angle *= _trackBallSettings.rotateSpeed;
      glm::quat _q = glm::angleAxis(angle, axis);

      glm::vec3 offset = _eye - _center;
      offset = _q * offset;
      _eye = _center + offset;

      _up = _q * _up;

      _updateViewMatrix();
    }

    _prevP = _currP;
  }

  // on mouse up
  if (_isRotating && ImGui::IsMouseReleased(_trackBallSettings.rotateButton)) {
    _isRotating = false;

    _updateViewMatrix();
  }
}

void TrackballCameraThree::_setCenter(glm::vec3 newCenter) {
  if (_center == newCenter)
    return;

  glm::vec3 direction = newCenter - _center;

  _center = newCenter;
  _eye += direction;

  _updateViewMatrix();
}

glm::vec2 TrackballCameraThree::_getMouseOnCircle(glm::vec2 localPosition) const {
  return {(localPosition.x - _width * 0.5) / (_width * 0.5), (_height + 2 * -localPosition.y) / _width};
}

void TrackballCameraThree::_controls() {}

void TrackballCameraThree::_updateViewMatrix() { setViewMatrix(_eye, _center, _up); }

[[nodiscard]] glm::vec3 TrackballCameraThree::eye() const { return _eye; }
[[nodiscard]] glm::vec3 TrackballCameraThree::center() const { return _center; }
[[nodiscard]] glm::vec3 TrackballCameraThree::up() const { return _up; }
