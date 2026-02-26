#ifndef IMGUI_OPENGL_HPP
#define IMGUI_OPENGL_HPP
#pragma once

#include <stack>

#include <imgui.h>
#include <imgui_internal.h>

#include <glad/gl.h>

#include "../gl/frameBufferHelper.hpp"

namespace ImGui {

inline ImPool<FrameBufferHelper> fboData; // NOLINT(cert-err58-cpp)
inline std::stack<ImGuiID> ID_stack;      // NOLINT(cert-err58-cpp)

inline bool BeginOpenGL(const char *str_id, const ImVec2 &size = ImVec2(0, 0), bool border = false,
                        ImGuiWindowFlags flags = 0) {

  int beginFlag = ImGui::BeginChild(str_id, size, border, flags);
  if (!beginFlag)
    return beginFlag;

  ID_stack.push(ImGui::GetID(str_id));

  FrameBufferHelper *data = fboData.GetOrAddByKey(ID_stack.top());
  data->bindDraw();

  const ImVec2 windowSize = ImGui::GetContentRegionAvail();
  data->onResize(static_cast<GLsizei>(windowSize.x), static_cast<GLsizei>(windowSize.y));

  return beginFlag;
}

inline void EndOpenGL() {

  FrameBufferHelper *data = fboData.GetByKey(ID_stack.top());
  FrameBufferHelper::unbindDraw();

  const float window_width = ImGui::GetContentRegionAvail().x;
  const float window_height = ImGui::GetContentRegionAvail().y;
  ImVec2 pos = ImGui::GetCursorScreenPos();

  ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)data->getTextureId(), ImVec2(pos.x, pos.y),
                                       ImVec2(pos.x + window_width, pos.y + window_height), ImVec2(0, 1),
                                       ImVec2(1, 0));

  ID_stack.pop();
  ImGui::EndChild();

  if (!ID_stack.empty())
    fboData.GetByKey(ID_stack.top())->bindDraw();
}

} // namespace ImGui

#endif // !IMGUI_OPENGL_HPP
