#ifndef CAMERA_HPP
#define CAMERA_HPP
#include "utils/utils.hpp"
#pragma once

#include <cstdio>

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

  inline explicit CameraSettings(float fov = 30.0f, float nearPlane = 0.1f, float farPlane = 100.0f,
                                 float cameraDistanceMin = 1.0f, float cameraDistanceMax = 40.0f,
                                 ImGuiMouseButton moveButton = ImGuiMouseButton_Middle)
      : fov(fov), nearPlane(nearPlane), farPlane(farPlane), cameraDistanceMin(cameraDistanceMin),
        cameraDistanceMax(cameraDistanceMax), moveButton(moveButton) {}
};

// TODO: rotate up
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

    // on mouse down
    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(_settings.moveButton)) {
      if (_moveMode != MoveMode::Pan && io.KeyShift) {
        _moveMode = MoveMode::Pan;
        _onPanStart(localMousePos);
      } else if (_moveMode != MoveMode::Rotate) {
        _moveMode = MoveMode::Rotate;
        _onRotateStart(localMousePos);
      }
    }

    // on mouse move
    if (ImGui::IsMouseDragging(_settings.moveButton)) {
      if (_moveMode == MoveMode::Pan && io.KeyShift) {
        _onPanMove(localMousePos);
      } else if (_moveMode == MoveMode::Rotate) {
        _onRotateMove(localMousePos);
      }
    }

    if (ImGui::IsMouseReleased(_settings.moveButton)) {
      if (_moveMode == MoveMode::Pan) {
        _onPanEnd();
      } else if (_moveMode == MoveMode::Rotate) {
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
  enum class MoveMode : int { None, Rotate, Pan, Zoom };
  MoveMode _moveMode = MoveMode::None;

  // zoom
  virtual inline void _zoom(float wheelDelta) {}

  // rotate
  virtual inline void _onRotateStart(const glm::vec2 &localMousePos) {}
  virtual inline void _onRotateMove(const glm::vec2 &localMousePos) {}
  virtual inline void _onRotateEnd() {}

  // pan
  glm::vec2 _anchorMousePos{};
  glm::vec3 _anchorCenter{};

  virtual inline void _onPanStart(const glm::vec2 &localMousePos) {
    _anchorMousePos = Utils::toGlm(ImGui::GetMousePos());
    _anchorCenter = center();
  }
  virtual inline void _onPanMove(const glm::vec2 &localMousePos) {
    ImVec2 mousePos = ImGui::GetMousePos();
    glm::vec2 mouseDelta = {mousePos.x - _anchorMousePos.x, mousePos.y - _anchorMousePos.y};

    // TODO: panning
    // _setCenter({});

    printf("delta: %f, %f\n", mouseDelta.x, mouseDelta.y);
  }
  virtual inline void _onPanEnd() {}

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
