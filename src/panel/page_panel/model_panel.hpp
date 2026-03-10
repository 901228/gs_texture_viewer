#ifndef MODEL_PANEL_HPP
#define MODEL_PANEL_HPP
#pragma once

#include <memory>
#include <string>

#include <glm/glm.hpp>

#include "page_panel.hpp"

class Model;
class Camera;
class TextureEditor;

class ModelPanel : public PagePanel {
public:
  ModelPanel();
  ~ModelPanel() override;

  inline std::string name() override { return "Model View"; }

protected:
  void _attach() override;
  void _detach() override;
  void _render() override;
  void _renderParameterization() override;
  void _onResize(float width, float height) override;
  void _controls() override;

private:
  // model
  std::unique_ptr<Model> model;
  std::unique_ptr<Camera> camera;

private:
  // render options
  bool wire = false;
  bool _renderSelectedOnly = false;

  enum class RenderingMode : int { Mesh, TextureCoords, Texture };
  RenderingMode _renderingMode = RenderingMode::Mesh;

private:
  // texture
  std::unique_ptr<TextureEditor> _textureEditor;

private:
  glm::vec3 _lightDir{0, -1, 0};
  float _lightIntensity = 1.0f;
};

#endif // !MODEL_PANEL_HPP
