#ifndef CAMERA_HPP
#define CAMERA_HPP
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <cstdio>

class CameraSettings {
public:
  float fov;

  float nearPlane;
  float farPlane;

  float cameraDistanceMin;
  float cameraDistanceMax;

  ImGuiMouseButton rotateButton;
  ImGuiMouseButton panButton;

  inline explicit CameraSettings(float fov = 30.0f, float nearPlane = 0.1f, float farPlane = 100.0f,
                                 float cameraDistanceMin = 1.0f, float cameraDistanceMax = 40.0f,
                                 ImGuiMouseButton rotateButton = ImGuiMouseButton_Middle,
                                 ImGuiMouseButton panButton = ImGuiMouseButton_Middle)
      : fov(fov), nearPlane(nearPlane), farPlane(farPlane), cameraDistanceMin(cameraDistanceMin),
        cameraDistanceMax(cameraDistanceMax), rotateButton(rotateButton), panButton(panButton) {}
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

  inline void handleInput(ImVec2 pos) {
    ImGuiIO &io = ImGui::GetIO();

    // handle zoom
    if (ImGui::IsWindowHovered()) {
      float wheelDelta = io.MouseWheel;

      if (wheelDelta != 0)
        _zoom(wheelDelta);
    }

    // cancel panning
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      if (_isPanning) {
        _isPanning = false;
      }
    }

    // handle pan
    if (io.KeyShift) {

      // on mouse down
      if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(_settings.panButton)) {
        _isPanning = true;
        _anchorMousePos = {io.MousePos.x, io.MousePos.y};
        _anchorCenter = center();
      }

      // on mouse move
      if (_isPanning && ImGui::IsMouseDragging(_settings.panButton)) {
        glm::vec2 mouseDelta = {io.MousePos.x - _anchorMousePos.x, io.MousePos.y - _anchorMousePos.y};

        // TODO: panning
        // _setCenter({});

        printf("delta: %f, %f\n", mouseDelta.x, mouseDelta.y);
      }

      if (_isPanning && ImGui::IsMouseReleased(_settings.panButton)) {
        _isPanning = false;
      }
    }
    // handle rotation
    else {
      _handleInput(pos);
    }
  }

  inline void setCenter(glm::vec3 newCenter) { _setCenter(newCenter); }

  inline void controls(const glm::vec3 &modelCenter) {
    ImGui::SeparatorText("Camera Option");
    {
      if (ImGui::Button("Focus on Model", {ImGui::GetContentRegionAvail().x, 0})) {
        setCenter(modelCenter);
      }

      _controls();
    }
    ImGui::NewLine();
  }

protected:
  virtual void _onResize(float width, float height) = 0;
  virtual void _zoom(float wheelDelta) = 0;
  virtual void _handleInput(ImVec2 pos) = 0;
  virtual void _setCenter(glm::vec3 newCenter) = 0;
  virtual void _controls() = 0;

protected:
  CameraSettings _settings;

protected:
  // projection matrix
  float _width = 800, _height = 800;
  glm::mat4 _projectionMatrix{};

protected:
  // view matrix
  glm::mat4 _viewMatrix{};

  inline void setViewMatrix(glm::vec3 eye, glm::vec3 center, glm::vec3 up) {
    _viewMatrix = glm::lookAt(eye, center, up);
  }

protected:
  // pan
  bool _isPanning = false;
  glm::vec2 _anchorMousePos{};
  glm::vec3 _anchorCenter{};

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
