file(GLOB src_files
  *.cpp
  model/*.cpp
  panel/*.cpp
  panel/page_panel/*.cpp

  utils/*.cpp
  utils/camera/*.cpp
  utils/gl/*.cpp
  utils/imgui/*.cpp
  utils/mesh/*.cpp
  utils/texture/*.cpp

  gaussian/model/*.cpp
  gaussian/view/*.cpp
  gaussian/view/renderer/*.cpp
)

# add src files to executable
add_executable(${PROJECT_NAME} ${src_files})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)

# link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
  glad glfw imgui
  OpenMeshCore OpenMeshTools
  eigen glm::glm nfd::nfd CLI11::CLI11 spdlog::spdlog tomlplusplus::tomlplusplus stb
  magic_enum::magic_enum IconsFont marching_cube
  GAUSSIAN_LIB)

# copy executable to config build directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_BINARY_DIR}/$<CONFIG>
  COMMAND ${CMAKE_COMMAND} -E copy
  $<TARGET_FILE:${PROJECT_NAME}>
  ${CMAKE_BINARY_DIR}/$<CONFIG>/
)
