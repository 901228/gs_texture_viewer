#ifndef MAIN_WINDOW_HPP
#define MAIN_WINDOW_HPP
#pragma once

#include <memory>
#include <vector>

#define GL_SILENCE_DEPRECATION
#include <glad/gl.h>

// glad should be included before glfw
#include <GLFW/glfw3.h>

#include <imgui.h>

class Panel;

class MainWindow {

public:
  // constructor
  MainWindow();
  ~MainWindow();

private:
  // main functions
  GLFWwindow *window = nullptr;
  bool Init();
  void Run();
  void Destroy();
  bool isReady = false;

private:
  // sub functions
  void CreateImGuiComponents();

  void CreateMenuBar();
  void CreateMainView();
  void CreateParameterizationPanel();
  void CreateControlPanel();
  void CreateSettingPage();

private:
  // constants
  const float windowWidth = 1000;
  const float windowHeight = 800;

public:
  // NOLINTBEGIN(hicpp-signed-bitwise)
  static const ImGuiWindowFlags flag = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                       ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar |
                                       ImGuiWindowFlags_NoBringToFrontOnFocus;
  static const ImGuiWindowFlags topFlag = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse |
                                          ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;
  // NOLINTEND(hicpp-signed-bitwise)

private:
  // variables for layout
  float menubarOffsetY = 20;
  float mainWidth = 760;
  float mainHeight = windowHeight;

  ImVec2 windowPos{0, 0};

private:
  // variables
  bool isSettingPageOpened = false;

  float sliderFloat = 0;

private:
  std::vector<std::unique_ptr<Panel>> panels;
  int currentPanel = 0;
};

#endif // !MAIN_WINDOW_HPP
