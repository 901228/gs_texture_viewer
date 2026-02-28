#ifndef PAGE_PANEL_HPP
#define PAGE_PANEL_HPP
#pragma once

#include "../panel.hpp"

class PagePanel : public Panel {
public:
  inline void renderParameterization() {
    attach();
    _renderParameterization();
  }

  inline void controls() {
    attach();
    _controls();
  }

protected:
  virtual void _renderParameterization() = 0;
  virtual void _controls() = 0;
};

#endif // !PAGE_PANEL_HPP
