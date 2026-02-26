#ifndef TRACKBALL_CAMERA_HPP
#define TRACKBALL_CAMERA_HPP
#pragma once

#include "camera.hpp"

class TrackballCameraSettings : public CameraSettings {
public:
  float radius;
  bool invertX;
  bool invertY;

  inline explicit TrackballCameraSettings(float fov = 30.0f, float nearPlane = 0.1f, float farPlane = 100.0f,
                                          float cameraDistanceMin = 1.0f, float cameraDistanceMax = 40.0f,
                                          float radius = 1.0f, bool invertX = true, bool invertY = false)
      : CameraSettings(fov, nearPlane, farPlane, cameraDistanceMin, cameraDistanceMax), radius(radius),
        invertX(invertX), invertY(invertY) {}
};

class TrackballCamera : public Camera {
public:
  explicit TrackballCamera(float cameraDistance,
                           TrackballCameraSettings settings = TrackballCameraSettings());

private:
  void _onResize(float width, float height) override;
  void _zoom(float wheelDelta) override;
  void _handleInput(ImVec2 pos) override;
  void _setCenter(glm::vec3 newCenter) override;
  void _controls() override;

  void _resetRotation();

private:
  bool _isRotating = false;
  glm::vec3 _startP{};

  glm::quat _lastQ{1, 0, 0, 0};
  glm::quat _currQ{1, 0, 0, 0};

  glm::vec3 _center;
  glm::vec3 _initialEye;
  float _cameraDistance;

private:
  void _updateViewMatrix();

private:
  TrackballCameraSettings _trackBallSettings;

public:
  [[nodiscard]] glm::mat4 rotationMatrix() const;

  [[nodiscard]] glm::vec3 eye() const override;
  [[nodiscard]] glm::vec3 center() const override;
  [[nodiscard]] glm::vec3 up() const override;
};

#endif // !TRACKBALL_CAMERA_HPP
