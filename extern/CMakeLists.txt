add_subdirectory(glad)
add_subdirectory(glfw)

# ImGui
add_library(imgui
  imgui/ImGui/imconfig.h
  imgui/ImGui/imgui.cpp
  imgui/ImGui/imgui.h
  imgui/ImGui/imgui_demo.cpp
  imgui/ImGui/imgui_draw.cpp
  imgui/ImGui/imgui_internal.h
  imgui/ImGui/imgui_tables.cpp
  imgui/ImGui/imgui_widgets.cpp
  imgui/ImGui/imstb_rectpack.h
  imgui/ImGui/imstb_textedit.h
  imgui/ImGui/imstb_truetype.h

  imgui/ImGui/backends/imgui_impl_glfw.cpp
  imgui/ImGui/backends/imgui_impl_glfw.h
  imgui/ImGui/backends/imgui_impl_opengl3.cpp
  imgui/ImGui/backends/imgui_impl_opengl3.h
  imgui/ImGui/backends/imgui_impl_opengl3_loader.h

  imgui/ImGuizmo/GraphEditor.cpp
  imgui/ImGuizmo/GraphEditor.h
  imgui/ImGuizmo/ImCurveEdit.cpp
  imgui/ImGuizmo/ImCurveEdit.h
  imgui/ImGuizmo/ImGradient.cpp
  imgui/ImGuizmo/ImGradient.h
  imgui/ImGuizmo/ImGuizmo.cpp
  imgui/ImGuizmo/ImGuizmo.h
  imgui/ImGuizmo/ImSequencer.cpp
  imgui/ImGuizmo/ImSequencer.h
  imgui/ImGuizmo/ImZoomSlider.h
)
target_compile_definitions(imgui PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
target_include_directories(imgui INTERFACE "imgui/")
target_include_directories(imgui PUBLIC "imgui/ImGui/") # imgui core headers (required by backends)
target_link_libraries(imgui PUBLIC glfw)

add_subdirectory(OpenMesh)
add_subdirectory(eigen)
add_subdirectory(glm)
add_subdirectory(nfd)
add_subdirectory(CLI11)
add_subdirectory(spdlog)
add_subdirectory(tomlplusplus)

add_library(stb
  stb/stb/stb_image.h
  stb/stb/stb_image.cpp
  stb/stb/stb_image_resize2.h
  stb/stb/stb_image_resize2.cpp
  stb/stb/stb_include.h
  stb/stb/stb_include.cpp
)
target_include_directories(stb PUBLIC "stb/")

add_subdirectory(magic_enum)
