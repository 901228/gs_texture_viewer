#include "trackball_camera.hpp"

namespace {
glm::vec2 normalizeMousePos(const ImVec2 &mousePos, float width, float height, bool invertX, bool invertY) {
  // normalize the mouse input pos
  // C = {1/2 * (width - 1), 1/2 * (height - 1)}
  // s = min(width, height) - 1
  // px =  2/s * (Px - Cx) =  1/s * (2Px -  width + 1)
  // py = -2/s * (Py - Cy) = -1/s * (2Py - height + 1)
  float s = std::min(width, height) - 1.0f;
  float px = invertX ? (width - 2.0f * mousePos.x + 1.0f) / s : (2.0f * mousePos.x - width + 1.0f) / s;
  float py = invertY ? -(height - 2.0f * mousePos.y + 1.0f) / s : -(2.0f * mousePos.y - height + 1.0f) / s;

  return {px, py};
}

glm::vec3 projectToSphere(const glm::vec2 &p, float radius) {

  float d = p.x * p.x + p.y * p.y;
  float squareR = radius * radius;

  float z;
  if (2 * d <= squareR) {
    // inside the hemisphere
    z = std::sqrt(squareR - d);
  } else {
    // outside the hemisphere
    z = (squareR / 2) / std::sqrt(d);
  }

  return {p, z};
}

glm::vec3 project(const ImVec2 &mousePos, float width, float height, float radius, bool invertX,
                  bool invertY) {
  return projectToSphere(normalizeMousePos(mousePos, width, height, invertX, invertY), radius);
}

glm::quat quatNormalize(float w, float x, float y, float z) {
  // We assume |Q| > 0 for internal usage
  float il = 1.0f / std::sqrt(w * w + x * x + y * y + z * z);

  return {w * il, x * il, y * il, z * il};
}

glm::quat fromVectors(glm::vec3 u, glm::vec3 v) {

  // Normalize u and v
  glm::vec3 u_normalize = glm::length(u) > 0 ? glm::normalize(u) : u;
  glm::vec3 v_normalize = glm::length(v) > 0 ? glm::normalize(v) : v;

  // Calculate dot product of normalized u and v
  float dot = glm::dot(u_normalize, v_normalize);

  // Parallel when dot > 0.999999
  if (dot >= 1 - std::numeric_limits<float>::epsilon()) {
    return {1, 0, 0, 0};
  }

  // Anti-Parallel (close to PI) when dot < -0.999999
  if (1 + dot <= std::numeric_limits<float>::epsilon()) {

    // Rotate 180° around any orthogonal vector
    // axis = len(cross([1, 0, 0], u)) == 0 ? cross([0, 1, 0], u) : cross([1,
    // 0, 0], u) and therefore
    //    return Quaternion['fromAxisAngle'](Math.abs(ux) > Math.abs(uz) ?
    //    [-uy, ux, 0] : [0, -uz, uy], Math.PI)
    // or return Quaternion['fromAxisAngle'](Math.abs(ux) > Math.abs(uz) ? [
    // uy,-ux, 0] : [0,  uz,-uy], Math.PI) or ...

    // Since fromAxisAngle(axis, PI) == Quaternion(0, axis).normalize(),
    if (std::abs(u_normalize.x) > std::abs(u_normalize.z)) {
      return quatNormalize(0, -u_normalize.y, u_normalize.x, 0);
    } else {
      return quatNormalize(0, 0, -u_normalize.z, u_normalize.y);
    }
  }

  // w = cross(u, v)
  glm::vec3 w = glm::cross(u_normalize, v_normalize);

  // |Q| = sqrt((1.0 + dot) * 2.0)
  return quatNormalize(1 + dot, w.x, w.y, w.z);
}

} // namespace

TrackballCamera::TrackballCamera(float cameraDistance, TrackballCameraSettings settings)
    : Camera(settings), _trackBallSettings(settings), _cameraDistance(std::abs(cameraDistance)),
      _center({0, 0, 0}), _initialEye({0, 0, cameraDistance}) {

  _updateViewMatrix();
}

void TrackballCamera::_onResize(float width, float height) {}

void TrackballCamera::_zoom(float wheelDelta) {

  // update distance
  float newDistance =
      glm::clamp(_cameraDistance - wheelDelta, _settings.cameraDistanceMin, _settings.cameraDistanceMax);
  glm::vec3 direction = glm::normalize(_initialEye - _center);

  _initialEye = _center + direction * newDistance;
  _cameraDistance = newDistance;
  _updateViewMatrix();
}

void TrackballCamera::_handleInput(ImVec2 pos) {
  ImVec2 mousePos = ImGui::GetMousePos();
  ImVec2 localMousePos = {mousePos.x - pos.x, mousePos.y - pos.y};

  if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
    if (_isRotating) {
      _isRotating = false;
      _currQ = {1.0f, 0.0f, 0.0f, 0.0f};
      _updateViewMatrix();
    }
  }

  // on mouse down
  if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(_trackBallSettings.rotateButton)) {
    _isRotating = true;
    _startP = project(localMousePos, _width, _height, _trackBallSettings.radius, _trackBallSettings.invertX,
                      _trackBallSettings.invertY);
  }

  // on mouse move
  if (_isRotating && ImGui::IsMouseDragging(_trackBallSettings.rotateButton)) {
    glm::vec3 q = project(localMousePos, _width, _height, _trackBallSettings.radius,
                          _trackBallSettings.invertX, _trackBallSettings.invertY);

    _currQ = fromVectors(_startP, q);

    _updateViewMatrix();
  }

  // on mouse up
  if (_isRotating && ImGui::IsMouseReleased(_trackBallSettings.rotateButton)) {
    _isRotating = false;
    _lastQ = _lastQ * _currQ;
    _currQ = {1.0f, 0.0f, 0.0f, 0.0f};
    _updateViewMatrix();
  }
}

void TrackballCamera::_setCenter(glm::vec3 newCenter) {
  if (_center == newCenter)
    return;

  glm::vec3 direction = newCenter - _center;

  _center = newCenter;
  _initialEye += direction;
  _updateViewMatrix();
}

void TrackballCamera::_controls() {
  if (ImGui::Button("Reset Camera Rotation", {ImGui::GetContentRegionAvail().x, 0})) {
    _resetRotation();
  }
}

void TrackballCamera::_resetRotation() {
  _lastQ = {1, 0, 0, 0};
  _currQ = {1, 0, 0, 0};
  _updateViewMatrix();
}

[[nodiscard]] glm::mat4 TrackballCamera::rotationMatrix() const {
  glm::quat rotation = _lastQ * _currQ;
  return glm::mat4_cast(rotation);
}

void TrackballCamera::_updateViewMatrix() { setViewMatrix(eye(), _center, up()); }

[[nodiscard]] glm::vec3 TrackballCamera::eye() const {
  return glm::vec3(rotationMatrix() * glm::vec4(_initialEye, 1.0f)) + _center;
}
[[nodiscard]] glm::vec3 TrackballCamera::center() const { return _center; }
[[nodiscard]] glm::vec3 TrackballCamera::up() const {
  return {rotationMatrix() * glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)};
}
