#ifndef GS_TEXTURE_PANEL_HPP
#define GS_TEXTURE_PANEL_HPP
#pragma once

#include <memory>
#include <string>

#include "page_panel.hpp"

#include "../gaussian/rasterizer/texture_rasterizer.hpp"
#include "../gaussian/view/gs_view.hpp"
#include "../utils/mesh/solve_uv.hpp"

class TextureEditor;
class TextureGaussianModel;
class Camera;
class FrameBufferHelper;

class TextureGSPanel : public PagePanel {
public:
  TextureGSPanel();
  ~TextureGSPanel() override;

  inline std::string name() override { return "GS Texture"; }

protected:
  void _attach() override;
  void _detach() override;
  void _render() override;
  void _renderParameterization() override;
  void _onResize(float width, float height) override;
  void _controls() override;

private:
  std::unique_ptr<Camera> camera;

private:
  // gaussian
  GaussianView::RenderingMode currMode = GaussianView::RenderingMode::Splats;
  std::unique_ptr<TextureGaussianModel> _textureGaussianModel;
  CudaRasterizer::RenderingMode _textureRenderMode = CudaRasterizer::RenderingMode::Texture;

private:
  // model
  std::unique_ptr<FrameBufferHelper> selectingFBO;

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

#endif // !GS_TEXTURE_PANEL_HPP
