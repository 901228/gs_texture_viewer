#ifndef IMGUI_REFERENCE_HPP
#define IMGUI_REFERENCE_HPP
#pragma once

#include <functional>

#include <ImGui/imgui.h>

namespace ImGui {

inline bool SliderFloat(const char *label, std::function<const float()> getter,
                        std::function<void(float)> setter, float min, float max, const char *format = "%.3f",
                        ImGuiSliderFlags flags = 0) {

  float val = getter();
  bool ret = ImGui::SliderFloat(label, &val, min, max, format, flags);
  if (ret)
    setter(val);
  return ret;
}

inline bool Checkbox(const char *label, std::function<const bool()> getter,
                     std::function<void(bool)> setter) {

  bool val = getter();
  bool ret = ImGui::Checkbox(label, &val);
  if (ret)
    setter(val);
  return ret;
}

}; // namespace ImGui

#endif // !IMGUI_REFERENCE_HPP
