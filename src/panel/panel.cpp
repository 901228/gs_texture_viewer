#include "panel.hpp"

void Panel::render() {

  checkInited();

  _render();
}

void Panel::renderParameterization() {

  checkInited();

  _renderParameterization();
}

void Panel::onResize(float width, float height) {
  if (this->width == width && this->height == height)
    return;

  checkInited();

  this->width = width;
  this->height = height;
  _onResize(width, height);
}

void Panel::controls() {

  checkInited();

  _controls();
}

void Panel::checkInited() {

  if (!inited) {
    _init();
    inited = true;
  }
}
