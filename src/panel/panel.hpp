#ifndef PANEL_HPP
#define PANEL_HPP
#pragma once

#include <cassert>
#include <string>

class Panel {
public:
  virtual inline ~Panel() { assert(!_attached && "Must call detach() before destroying Panel"); }

public:
  inline void render() {
    attach();
    _render();
  }
  inline void onResize(float width, float height) {
    if (_width == width && _height == height)
      return;

    _width = width;
    _height = height;
    attach();
    _onResize(width, height);
  }

  virtual inline std::string name() = 0;

protected:
  virtual inline void _attach() {}
  // TODO: finish the attach/detach logic
  virtual inline void _detach() {}
  virtual void _render() = 0;
  virtual void _onResize(float width, float height) = 0;

protected:
  float _width = 1, _height = 1;

protected:
  bool _attached = false;

  void attach() {
    if (!_attached) {
      _attach();
      _attached = true;
    }
  }
  void detach() {
    if (_attached) {
      _detach();
      _attached = false;
    }
  }
};

#endif // !PANEL_HPP
