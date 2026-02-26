file(GLOB src_files *.cpp
  model/*.cpp
  panel/*.cpp
  utils/camera/*.cpp
  utils/gl/*.cpp
  utils/imgui/*.cpp
  utils/mesh/*.cpp
  utils/texture/*.cpp)

add_subdirectory(gaussian)

# add src files to executable
add_executable(${PROJECT_NAME} ${src_files})

target_link_libraries(${PROJECT_NAME} PRIVATE
  glad glfw imgui
  OpenMeshCore OpenMeshTools eigen glm nfd CLI11::CLI11 spdlog::spdlog tomlplusplus::tomlplusplus stb
  GAUSSIAN_LIB)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_BINARY_DIR}/$<CONFIG>
  COMMAND ${CMAKE_COMMAND} -E copy
  $<TARGET_FILE:${PROJECT_NAME}>
  ${CMAKE_BINARY_DIR}/$<CONFIG>/
)

# post processing
if(WIN32)
  # Windows
  add_custom_command(TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${PROJECT_SOURCE_DIR}/lib/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    VERBATIM)
elseif(UNIX AND NOT APPLE)
# Linux
elseif(APPLE)
  # Apple
endif()
