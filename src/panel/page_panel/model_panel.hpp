#ifndef MODEL_PANEL_HPP
#define MODEL_PANEL_HPP
#pragma once

#include <memory>
#include <string>

#include <glm/glm.hpp>

#include "page_panel.hpp"

#include "../utils/mesh/solve_uv.hpp"

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

  enum class RenderingMode : int { Mesh, TextureCoords, Texture };
  RenderingMode _renderingMode = RenderingMode::Mesh;

private:
  // editing options
  int brushRadius = 10;

  SolveUV::SolvingMode _solvingMode = SolveUV::SolvingMode::ExpMap;
  bool _solved = false;

private:
  // texture
  bool _editingTexture = false;
  std::unique_ptr<TextureEditor> _textureEditor;
};

#endif // !MODEL_PANEL_HPP
