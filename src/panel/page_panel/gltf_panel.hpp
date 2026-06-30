#ifndef GLTF_PANEL_HPP
#define GLTF_PANEL_HPP
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "page_panel.hpp"
#include "utils/light.hpp"

class GltfModel;
class Camera;
class TextureEditor;

// Panel for viewing a .glb scene: renders every sub-mesh with its PBR material under
// directional lighting, and lets the user paint a decal texture onto one sub-mesh.
class GltfPanel : public PagePanel {
public:
  GltfPanel();
  ~GltfPanel() override;

  inline std::string name() override { return "GLB View"; }

protected:
  void _attach() override;
  void _detach() override;
  void _render() override;
  void _renderParameterization() override;
  void _onResize(float width, float height) override;
  void _controls() override;

private:
  std::unique_ptr<GltfModel> model;
  std::unique_ptr<Camera> camera;

private:
  bool wire = false;
  bool _renderSelectedOnly = false;
  bool _flipNormals = false;

  enum class RenderingMode : int { Mesh, TextureCoords, Texture };
  RenderingMode _renderingMode = RenderingMode::Mesh;

private:
  std::unique_ptr<TextureEditor> _textureEditor;

private:
  std::vector<Light> _lights = defaultLights();
  bool _showLights = true;

  // Light orbit animation: while enabled, an azimuth offset advances over time so the
  // whole rig circles the model (handy for reviewing a PBR texture from every angle).
  bool _animateOrbit = false;
  float _orbitSpeed = 60.0f; // degrees per second
  float _orbitAngle = 0.0f;  // current accumulated offset, degrees

  // A reasonable three-point-ish lighting rig used as the starting setup.
  static std::vector<Light> defaultLights();
  // _lights with the current orbit offset applied to each light's azimuth.
  std::vector<Light> effectiveLights() const;
  // Draw a 2D overlay gizmo (sun marker + incoming-direction arrow) for each enabled
  // light in `lights`, projected into the 3D view whose top-left is (originX, originY).
  void renderLightGizmos(float originX, float originY, const std::vector<Light> &lights);
};

#endif // !GLTF_PANEL_HPP
