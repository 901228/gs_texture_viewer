add_subdirectory(glad)
add_subdirectory(glfw)

# ImGui
add_library(imgui
  imgui/imconfig.h
  imgui/imgui.cpp
  imgui/imgui.h
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_internal.h
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/imstb_rectpack.h
  imgui/imstb_textedit.h
  imgui/imstb_truetype.h

  imgui/backends/imgui_impl_glfw.cpp
  imgui/backends/imgui_impl_glfw.h
  imgui/backends/imgui_impl_opengl3.cpp
  imgui/backends/imgui_impl_opengl3.h
  imgui/backends/imgui_impl_opengl3_loader.h

  imguizmo/GraphEditor.cpp
  imguizmo/GraphEditor.h
  imguizmo/ImCurveEdit.cpp
  imguizmo/ImCurveEdit.h
  imguizmo/ImGradient.cpp
  imguizmo/ImGradient.h
  imguizmo/ImGuizmo.cpp
  imguizmo/ImGuizmo.h
  imguizmo/ImSequencer.cpp
  imguizmo/ImSequencer.h
  imguizmo/ImZoomSlider.h
)
target_compile_definitions(imgui PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
target_include_directories(imgui PUBLIC "imgui/")
target_include_directories(imgui PUBLIC "imgui/backends/")
target_include_directories(imgui PUBLIC "imguizmo")

add_subdirectory(OpenMesh)
add_subdirectory(eigen)
add_subdirectory(glm)
add_subdirectory(nfd)
add_subdirectory(CLI11)
add_subdirectory(spdlog)
add_subdirectory(tomlplusplus)

add_library(stb
  stb/stb_image.h
  stb/stb_image.cpp
  stb/stb_image_resize2.h
  stb/stb_image_resize2.cpp
)
target_include_directories(stb PUBLIC "stb/")

add_subdirectory(magic_enum)
