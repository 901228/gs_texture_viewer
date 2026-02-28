#ifndef GAUSSIAN_PANEL_HPP
#define GAUSSIAN_PANEL_HPP
#pragma once

#include <memory>
#include <string>

#include "page_panel.hpp"

#include "../gaussian/view/gs_view.hpp"

class Camera;

class GaussianPanel : public PagePanel {
public:
  GaussianPanel();
  ~GaussianPanel() override;

  inline std::string name() override { return "Gaussian View"; }

protected:
  void _attach() override;
  void _detach() override;
  void _render() override;
  void _renderParameterization() override;
  void _onResize(float width, float height) override;
  void _controls() override;

private:
  // gaussian
  std::unique_ptr<Camera> camera;

  GaussianView::RenderingMode currMode = GaussianView::RenderingMode::Splats;
  std::unique_ptr<GaussianModel> _gsModel;
};

#endif // !GAUSSIAN_PANEL_HPP
