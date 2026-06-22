#ifndef GS_TEXTURE_MESH_PANEL_HPP
#define GS_TEXTURE_MESH_PANEL_HPP
#pragma once

#include <memory>
#include <string>

#include "page_panel.hpp"

#include "gaussian/view/gs_view.hpp"

#include "rasterizer/defines.hpp"

class TextureEditor;
class TextureGaussianModelMesh;
class Camera;

class TextureGSMeshPanel : public PagePanel {
public:
  TextureGSMeshPanel();
  ~TextureGSMeshPanel() override;

  inline std::string name() override { return "GS Texture Mesh"; }

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
  std::unique_ptr<TextureGaussianModelMesh> _textureGaussianModel;
  CudaRasterizer::MaskCullingMode _maskCullingMode = CudaRasterizer::MaskCullingMode::Mixed_ND;

private:
  // texture
  std::unique_ptr<TextureEditor> _textureEditor;

  glm::vec3 _lightDir{0, -1, 0};
  float _lightIntensity = 1.0f;
};

#endif // !GS_TEXTURE_MESH_PANEL_HPP
