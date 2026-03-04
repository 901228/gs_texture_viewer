#include "trackball_camera_three.hpp"

namespace {

bool _insideSphere(glm::vec2 p) { return p.x * p.x + p.y * p.y <= 1.0f; }

glm::vec3 _projectToSphere(glm::vec2 p) {
  glm::vec3 v(p.x, p.y, 0.0f);

  if (_insideSphere(v)) {
    v.z = std::sqrtf(1.0f - p.x * p.x - p.y * p.y);
  } else {
    v = glm::normalize(v);
  }

  return glm::normalize(v);
}

} // namespace

TrackballCameraThree::TrackballCameraThree(float cameraDistance, TrackballCameraThreeSettings settings)
    : Camera(settings), _eye({0, 0, cameraDistance}) {

  _updateViewMatrix();
}

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

void TrackballCameraThree::_onRotateStart(const glm::vec2 &ndcMousePos) {
  _currP = ndcMousePos;
  _prevP = _currP;
}
void TrackballCameraThree::_onRotateMove(const glm::vec2 &ndcMousePos) {

  _prevP = _currP;
  _currP = ndcMousePos;

  glm::vec2 _cursorDirection = _currP - _prevP;
  float angle = glm::length(_cursorDirection);

  if (angle != 0) {

    // forward is from eye to center, so this is backward
    glm::vec3 backward = glm::normalize(_eye - _center);
    glm::vec3 right = glm::normalize(glm::cross(_up, backward));

    glm::vec3 up = glm::cross(backward, right);
    if (glm::length(up) < 1e-6f)
      _up = glm::vec3(0, 1, 0);
    else {
      _up = glm::normalize(up);
    }

    up = _up * _cursorDirection.y;
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
void TrackballCameraThree::_onRotateUpMove(const glm::vec2 &ndcMousePos) {

  _prevP = _currP;
  _currP = ndcMousePos;

  glm::vec2 _cursorDirection = _currP - _prevP;
  float angle = glm::length(_cursorDirection);

  if (angle != 0) {

    glm::vec3 forward = glm::normalize(_center - _eye);
    glm::vec3 right = normalize(cross(forward, _up));
    glm::vec3 up = glm::cross(right, forward);
    if (glm::length(up) < 1e-6f)
      _up = glm::vec3(0, 1, 0);
    else {
      _up = glm::normalize(up);
    }

    if (_prevP.x * _currP.y - _prevP.y * _currP.x < 0)
      angle = -angle;
    glm::quat q = glm::angleAxis(angle, forward);
    _up = q * _up;

    _updateViewMatrix();
  }

  _prevP = _currP;
}
void TrackballCameraThree::_onRotateEnd() { _updateViewMatrix(); }

void TrackballCameraThree::_setCenter(const glm::vec3 &newCenter) {
  if (_center == newCenter)
    return;

  glm::vec3 direction = newCenter - _center;

  _center = newCenter;
  _eye += direction;

  _updateViewMatrix();
}

void TrackballCameraThree::_updateViewMatrix() { setViewMatrix(_eye, _center, _up); }

[[nodiscard]] glm::vec3 TrackballCameraThree::eye() const { return _eye; }
[[nodiscard]] glm::vec3 TrackballCameraThree::center() const { return _center; }
[[nodiscard]] glm::vec3 TrackballCameraThree::up() const { return _up; }
