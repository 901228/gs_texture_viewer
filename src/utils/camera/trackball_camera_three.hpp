/**
 * Reference: https://github.com/mrdoob/three.js/blob/dev/examples/jsm/controls/TrackballControls.js
 */

#ifndef TRACKBALL_CAMERA_THREE_HPP
#define TRACKBALL_CAMERA_THREE_HPP
#pragma once

#include "camera.hpp"

class TrackballCameraThreeSettings : public CameraSettings {
public:
  float rotateSpeed;

  inline explicit TrackballCameraThreeSettings(float fov = 30.0f, float nearPlane = 0.1f,
                                               float farPlane = 100.0f, float cameraDistanceMin = 1.0f,
                                               float cameraDistanceMax = 40.0f,
                                               ImGuiMouseButton moveButton = ImGuiMouseButton_Middle,
                                               float rotateSpeed = 1.0f)
      : CameraSettings(fov, nearPlane, farPlane, cameraDistanceMin, cameraDistanceMax, moveButton),
        rotateSpeed(rotateSpeed) {}
};

class TrackballCameraThree : public Camera {
public:
  explicit TrackballCameraThree(float cameraDistance,
                                TrackballCameraThreeSettings settings = TrackballCameraThreeSettings());

private:
  void _zoom(float wheelDelta) override;
  void _setCenter(const glm::vec3 &newCenter) override;

  void _onRotateStart(const glm::vec2 &localMousePos) override;
  void _onRotateMove(const glm::vec2 &localMousePos) override;
  void _onRotateEnd() override;

private:
  glm::vec2 _currP;
  glm::vec2 _prevP;

  glm::vec3 _eye;
  glm::vec3 _center{0, 0, 0};
  glm::vec3 _up{0, 1, 0};

private:
  glm::vec2 _getMouseOnCircle(const glm::vec2 &localPosition) const;
  void _updateViewMatrix();

private:
  TrackballCameraThreeSettings _trackBallSettings;

public:
  [[nodiscard]] glm::vec3 eye() const override;
  [[nodiscard]] glm::vec3 center() const override;
  [[nodiscard]] glm::vec3 up() const override;
};

#endif // !TRACKBALL_CAMERA_THREE_HPP
