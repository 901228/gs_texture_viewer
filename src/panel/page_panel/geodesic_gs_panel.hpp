#ifndef GEODESIC_GS_TEXTURE_PANEL_HPP
#define GEODESIC_GS_TEXTURE_PANEL_HPP
#pragma once

#include <memory>
#include <string>

#include "page_panel.hpp"

#include "gaussian/view/gs_view.hpp"
#include "utils/mesh/isosurface.hpp"

#include "rasterizer/defines.hpp"

class TextureEditor;
class GeodesicGaussianModel;
class Camera;

class GeodesicTextureGSPanel : public PagePanel {
public:
  GeodesicTextureGSPanel();
  ~GeodesicTextureGSPanel() override;

  inline std::string name() override { return "Geodesic GS Texture"; }

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
  std::unique_ptr<GeodesicGaussianModel> _model;
  CudaRasterizer::MaskCullingMode _maskCullingMode = CudaRasterizer::MaskCullingMode::Mixed_ND;

private:
  // texture
  std::unique_ptr<TextureEditor> _textureEditor;

  glm::vec3 _lightDir{0, -1, 0};
  float _lightIntensity = 1.0f;

private:
  enum class ViewMode : int { Model, Isosurface };
  ViewMode _viewMode = ViewMode::Model;
  Isosurface::MarchingCubesResult _mc;
};

#endif // !GEODESIC_GS_TEXTURE_PANEL_HPP
