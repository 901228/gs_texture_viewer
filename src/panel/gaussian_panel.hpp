#ifndef GAUSSIAN_PANEL_HPP
#define GAUSSIAN_PANEL_HPP
#pragma once

#include <memory>
#include <string>

#include "../gaussian/view/gs_view.hpp"
#include "panel.hpp"

class Camera;

class GaussianPanel : public Panel {
public:
  GaussianPanel();
  ~GaussianPanel();

  inline std::string name() override { return "Gaussian View"; }

protected:
  void _init() override;
  void _render() override;
  void _renderParameterization() override;
  void _onResize(float width, float height) override;
  void _controls() override;

private:
  // gaussian
  std::unique_ptr<GaussianView> gaussianView;
  std::unique_ptr<Camera> camera;
};

#endif // !GAUSSIAN_PANEL_HPP
