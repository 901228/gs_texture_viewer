#ifndef GLTF_PANEL_HPP
#define GLTF_PANEL_HPP
#pragma once

#include <memory>
#include <string>

#include <glm/glm.hpp>

#include "page_panel.hpp"

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

  enum class RenderingMode : int { Mesh, TextureCoords, Texture };
  RenderingMode _renderingMode = RenderingMode::Mesh;

private:
  std::unique_ptr<TextureEditor> _textureEditor;

private:
  glm::vec3 _lightDir{0, -1, 0};
  float _lightIntensity = 1.0f;
};

#endif // !GLTF_PANEL_HPP
