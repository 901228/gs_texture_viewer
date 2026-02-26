#ifndef IMGUIZMO_CAMERA_HPP
#define IMGUIZMO_CAMERA_HPP
#pragma once

#include "camera.hpp"

class ImGuizmoCameraSettings : public CameraSettings {
public:
  inline explicit ImGuizmoCameraSettings(float fov = 30.0f, float nearPlane = 0.1f, float farPlane = 100.0f,
                                         float cameraDistanceMin = 1.0f, float cameraDistanceMax = 40.0f,
                                         ImGuiMouseButton rotateButton = ImGuiMouseButton_Middle,
                                         ImGuiMouseButton panButton = ImGuiMouseButton_Middle)
      : CameraSettings(fov, nearPlane, farPlane, cameraDistanceMin, cameraDistanceMax, rotateButton,
                       panButton) {}
};

class ImGuizmoCamera : public Camera {
public:
  explicit ImGuizmoCamera(float cameraDistance, ImGuizmoCameraSettings settings = ImGuizmoCameraSettings());

private:
  void _onResize(float width, float height) override;
  void _zoom(float wheelDelta) override;
  void _handleInput(ImVec2 pos) override;
  void _setCenter(glm::vec3 newCenter) override;
  void _controls() override;

private:
  void _setViewMatrix(glm::vec3 eye, glm::vec3 center, glm::vec3 up);

private:
  float _cameraDistance;
  float _cameraView[16]{};
  float _cameraProjection[16]{};
  ImVec2 gizmoSize{128, 128};

public:
  [[nodiscard]] glm::vec3 eye() const override;
  [[nodiscard]] glm::vec3 center() const override;
  [[nodiscard]] glm::vec3 up() const override;
  [[nodiscard]] glm::vec3 forward() const;
};

#endif // !IMGUIZMO_CAMERA_HPP
