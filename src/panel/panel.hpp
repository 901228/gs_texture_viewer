#ifndef PANEL_HPP
#define PANEL_HPP
#pragma once

#include <string>

#include <imgui.h>

class Panel {
public:
  void render();
  void renderParameterization();
  void onResize(float width, float height);
  void controls();
  virtual inline std::string name() = 0;

protected:
  virtual void _init() = 0;
  virtual void _render() = 0;
  virtual void _renderParameterization() = 0;
  virtual void _onResize(float width, float height) = 0;
  virtual void _controls() = 0;

protected:
  ImVec2 gizmoSize{128, 128};
  float width = 800, height = 800;

private:
  bool inited = false;
  void checkInited();
};

#endif // !PANEL_HPP
