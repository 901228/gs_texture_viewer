#ifndef CAMERA_HPP
#define CAMERA_HPP
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <ImGui/imgui.h>

class CameraSettings {
public:
  float fov;

  float nearPlane;
  float farPlane;

  float cameraDistanceMin;
  float cameraDistanceMax;

  ImGuiMouseButton moveButton;

  bool canRotateUp;
  float rotateUpRadius;

  float panSpeed;

  inline explicit CameraSettings(float fov = 30.0f, float nearPlane = 0.1f, float farPlane = 100.0f,
                                 float cameraDistanceMin = 1.0f, float cameraDistanceMax = 40.0f,
                                 ImGuiMouseButton moveButton = ImGuiMouseButton_Middle,
                                 bool canRotateUp = false, float rotateUpRadius = 0.8f, float panSpeed = 2.0f)
      : fov(fov), nearPlane(nearPlane), farPlane(farPlane), cameraDistanceMin(cameraDistanceMin),
        cameraDistanceMax(cameraDistanceMax), moveButton(moveButton), canRotateUp(canRotateUp),
        rotateUpRadius(rotateUpRadius), panSpeed(panSpeed) {}
};

class Camera {
public:
  inline explicit Camera(CameraSettings &settings) : _settings(settings) {}

  inline void onResize(float width, float height) {
    if (_width == width && _height == height)
      return;

    _width = width;
    _height = height;

    _projectionMatrix = glm::perspective(fov(), aspect(), _settings.nearPlane, _settings.farPlane);
    _onResize(width, height);
  }

  virtual inline void handleInput(const ImVec2 &pos) {
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

  inline void setCenter(const glm::vec3 &newCenter) { _setCenter(newCenter); }

  inline void controls(const glm::vec3 &modelCenter) {
    if (ImGui::CollapsingHeader("Camera Option")) {
      ImGui::Indent();

      if (ImGui::Button("Focus on Model", {ImGui::GetContentRegionAvail().x, 0})) {
        setCenter(modelCenter);
      }

      _controls();

      ImGui::Unindent();
    }
  }

protected:
  virtual inline void _onResize(float width, float height) {}
  virtual inline void _controls() {}
  virtual void _setCenter(const glm::vec3 &newCenter) = 0;

protected:
  CameraSettings _settings;

protected:
  // projection matrix
  float _width = 800, _height = 800;
  glm::mat4 _projectionMatrix{};

protected:
  // view matrix
  glm::mat4 _viewMatrix{};

  inline void setViewMatrix(const glm::vec3 &eye, const glm::vec3 &center, const glm::vec3 &up) {
    _viewMatrix = glm::lookAt(eye, center, up);
  }

protected:
  // mode
  enum class MoveMode : int { None, Rotate, RotateUp, Pan, Zoom };
  MoveMode _moveMode = MoveMode::None;

  inline glm::vec2 _getNDCPos(const glm::vec2 &localPos) const {
    return {
        (localPos.x / _width) * 2.0f - 1.0f,
        1.0f - (localPos.y / _height) * 2.0f // y should be flipped
    };
  }
  inline glm::vec2 _getlocalPosFromNDC(const glm::vec2 &ndcPos) const {
    return {
        _width + 2.0f * ndcPos.x + 2.0f,
        _height - 2.0f * ndcPos.y + 2.0f // y should be flipped
    };
  }
  static inline const bool _insideSphere(glm::vec2 p, float radius = 1.0f) {
    return p.x * p.x + p.y * p.y <= radius * radius;
  }

  // zoom
  virtual inline void _zoom(float wheelDelta) {}

  // rotate
  virtual inline void _onRotateStart(const glm::vec2 &ndcMousePos) {}
  virtual inline void _onRotateMove(const glm::vec2 &ndcMousePos) {}
  virtual inline void _onRotateEnd() {}
  virtual inline void _onRotateUpMove(const glm::vec2 &ndcMousePos) {}

  // pan
  glm::vec2 _anchorMousePos{};
  glm::vec3 _anchorCenter{};
  glm::vec3 _panUp{};
  glm::vec3 _panLeft{};

  virtual inline void _onPanStart(const glm::vec2 &ndcMousePos) {
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
  virtual inline void _onPanMove(const glm::vec2 &ndcMousePos) {
    glm::vec2 mouseDelta = (ndcMousePos - _anchorMousePos) * _settings.panSpeed;
    _setCenter(_anchorCenter + _panLeft * mouseDelta.x - _panUp * mouseDelta.y);
  }
  virtual inline void _onPanEnd() {
    _anchorMousePos = {};
    _panUp = {};
    _panLeft = {};
  }

public:
  [[nodiscard]] inline glm::mat4 viewMatrix() const { return _viewMatrix; }
  [[nodiscard]] inline const float *viewMatrixPointer() const { return glm::value_ptr(_viewMatrix); }
  [[nodiscard]] inline glm::mat4 projectionMatrix() const { return _projectionMatrix; }
  [[nodiscard]] inline const float *projectionMatrixPointer() const {
    return glm::value_ptr(_projectionMatrix);
  }

  [[nodiscard]] inline float fov(bool inDegree = false) const {
    return inDegree ? _settings.fov : glm::radians(_settings.fov);
  }
  [[nodiscard]] inline float aspect() const { return _width / _height; }

  [[nodiscard]] virtual glm::vec3 eye() const = 0;
  [[nodiscard]] virtual glm::vec3 center() const = 0;
  [[nodiscard]] virtual glm::vec3 up() const = 0;
};

#endif // !CAMERA_HPP
