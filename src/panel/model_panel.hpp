#ifndef MODEL_PANEL_HPP
#define MODEL_PANEL_HPP
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <nfd.h>

#include "panel.hpp"

#include "../utils/mesh/solve_uv.hpp"
#include "../utils/texture/texture.hpp"

class Model;
class FrameBufferHelper;
class Camera;

class ModelPanel : public Panel {
public:
  ModelPanel();
  ~ModelPanel();

  inline std::string name() override { return "Model View"; }

protected:
  void _init() override;
  void _render() override;
  void _renderParameterization() override;
  void _onResize(float width, float height) override;
  void _controls() override;

private:
  // model
  std::unique_ptr<Model> model;
  std::unique_ptr<FrameBufferHelper> selectingFBO;
  std::unique_ptr<Camera> camera;

  constexpr static std::string_view textureListPath = "textures.toml";

private:
  // render options
  bool wire = false;

  enum class RenderingMode : int { Mesh, TextureCoords, Texture };
  RenderingMode _renderingMode = RenderingMode::Mesh;
  bool _editingTexture = false;

private:
  // editing options
  int brushRadius = 10;

  SolveUV::SolvingMode _solvingMode = SolveUV::SolvingMode::ExpMap;
  bool _solved = false;

private:
  // texture
  std::vector<std::unique_ptr<ImageTexture>> _textureList{};
  int _selectedTexture = -1;

  // scale
  float _editingTextureScale = 1.0f;
  constexpr const static float _editingTextureScaleMin = 0.1f;
  constexpr const static float _editingTextureScaleStep = 0.1f;
  constexpr const static float _editingTextureScaleMax = 2.0f;

  // move
  glm::vec2 _editingTextureOffset{};
  glm::vec2 _editingTextureOffsetAnchor{};
  bool _isMouseLeftDown = false;
  glm::vec2 _mousePosAnchor{};

  // rotate
  float _editingTextureTheta = 0;

  enum TextureEditingMode { None, Rotate, Scale, Move };
  TextureEditingMode _textureEditingMode = TextureEditingMode::None;
};

#endif // !MODEL_PANEL_HPP
