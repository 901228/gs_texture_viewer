#ifndef GS_TEXTURE_PANEL_HPP
#define GS_TEXTURE_PANEL_HPP
#pragma once

#include <memory>
#include <string>

#include "panel.hpp"

#include "../gaussian/view/texture_gs_view.hpp"
#include "../utils/mesh/solve_uv.hpp"
#include "../utils/texture/texture.hpp"

class Camera;
class FrameBufferHelper;

class TextureGSPanel : public Panel {
public:
  TextureGSPanel();
  ~TextureGSPanel();

  inline std::string name() override { return "GS Texture"; }

protected:
  void _init() override;
  void _render() override;
  void _renderParameterization() override;
  void _onResize(float width, float height) override;
  void _controls() override;

private:
  void _handleSelectMesh();

private:
  // gaussian
  std::unique_ptr<TextureGSView> textureGaussian;
  std::unique_ptr<Camera> camera;

  constexpr static std::string_view textureListPath = "textures.toml";

private:
  // model
  std::unique_ptr<FrameBufferHelper> selectingFBO;

private:
  // editing options
  int brushRadius = 10;

  SolveUV::SolvingMode _solvingMode = SolveUV::SolvingMode::ExpMap;
  bool _solved = false;

private:
  bool renderSelectedOnly = false;

private:
  // texture
  std::vector<std::unique_ptr<ImageTexture>> _textureList{};
  int _selectedTexture = -1;
  bool _editingTexture = false;

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

  enum class TextureEditingMode : int { None, Rotate, Scale, Move };
  TextureEditingMode _textureEditingMode = TextureEditingMode::None;
};

#endif // !GS_TEXTURE_PANEL_HPP
