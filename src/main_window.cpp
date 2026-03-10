#include "main_window.hpp"

#include <glad/gl.h>

#include <ImGui/backends/imgui_impl_glfw.h>
#include <ImGui/backends/imgui_impl_opengl3.h>

#include <ImGui/imgui.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <nfd.h>
#include <stb/stb_image.h>

#include "panel/page_panel/gaussian_panel.hpp"
#include "panel/page_panel/model_panel.hpp"
#include "panel/page_panel/texture_gs_panel.hpp"
#include "utils/imgui/opengl.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"

MainWindow::MainWindow(bool isMultiViewport) {

  isReady = Init(isMultiViewport);
  if (isReady)
    Run();
}

MainWindow::~MainWindow() {

  if (isReady)
    Destroy();
}

static void glfw_error_callback(int error, const char *description) {

  ERROR("GLFW Error {}: {}", error, description);
}

bool MainWindow::Init(bool isMultiViewport) {

  // glfw initialization
  {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {

      ERROR("glfw init failed");
      return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // glfw window creation
    window = glfwCreateWindow(static_cast<int>(windowWidth), static_cast<int>(windowHeight),
                              "OpenGL with ImGui", nullptr, nullptr);
    if (window == nullptr) {

      ERROR("Failed to create GLFW window");
      glfwTerminate();
      return false;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // load icon
    GLFWimage images[1];
    images[0].pixels = stbi_load(Utils::Path::getAssetsPath("icons/icon.png").c_str(), &images[0].width,
                                 &images[0].height, nullptr, 4);
    if (images[0].pixels) {
      glfwSetWindowIcon(window, 1, images);
      stbi_image_free(images[0].pixels);
    }
  }

  // glad initialization
  if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {

    ERROR("Failed to initialize GLAD");
    glfwDestroyWindow(window);
    glfwTerminate();
    return false;
  }

  // GL initiation
  {
    GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    INFO("OpenGL {}.{}", major, minor);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // glEnable(GL_PROGRAM_POINT_SIZE);

    glPatchParameteri(GL_PATCH_VERTICES, 3);
  }

  // ImGui initialization
  {
    // if the platform is wayland, disable multi-viewports (glfw for wayland is not yet supported)
    const char *wayland_display = getenv("WAYLAND_DISPLAY");
    const char *session_type = getenv("XDG_SESSION_TYPE");
    bool is_wayland = (wayland_display != nullptr) || (session_type && strcmp(session_type, "wayland") == 0);
    isMultiViewport = !is_wayland && isMultiViewport;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    // NOLINTBEGIN(hicpp-signed-bitwise)
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
    if (isMultiViewport)
      io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable MultiViewports
    // NOLINTEND(hicpp-signed-bitwise)

    // Setup Dear ImGui style
    ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {

      ERROR("Failed to ImGui_ImplGlfw_InitForOpenGL");
      glfwDestroyWindow(window);
      glfwTerminate();
      return false;
    }
    if (!ImGui_ImplOpenGL3_Init("#version 430")) {

      ERROR("Failed to ImGui_ImplOpenGL3_Init");
      glfwDestroyWindow(window);
      glfwTerminate();
      return false;
    }
  }

  // nfd
  {
    NFD_Init();
  }

  // panels
  {
    panels.push_back(std::make_unique<TextureGSPanel>());
    panels.push_back(std::make_unique<ModelPanel>());
    panels.push_back(std::make_unique<GaussianPanel>());
  }

  return true;
}

void MainWindow::Run() {

  // render
  while (!glfwWindowShouldClose(window)) {

    glfwPollEvents();

    _frameRate = ImGui::GetIO().Framerate;

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui components
    CreateImGuiComponents();
    ImGui::Render();

    glViewport(0, 0, static_cast<GLsizei>(windowWidth), static_cast<GLsizei>(windowHeight));
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {

      GLFWwindow *backup_current_context = glfwGetCurrentContext();
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
      glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(window);
  }
}

void MainWindow::Destroy() {

  // OnDestroy
  panels.clear();
  ImGui::ClearOpenGL();

  NFD_Quit();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}

void MainWindow::CreateImGuiComponents() {

  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  windowPos = ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable
                  ? ImVec2(ImGui::GetMainViewport()->Pos)
                  : ImVec2(0, 0);

  CreateMenuBar();
  CreateMainView();
  CreateParameterizationPanel();
  CreateControlPanel();
  CreateSettingPage();
}

void MainWindow::CreateMenuBar() {

  if (ImGui::BeginMainMenuBar()) {

    menubarOffsetY = ImGui::GetWindowHeight();
    mainHeight = windowHeight - menubarOffsetY;

    if (ImGui::MenuItem("Settings")) {

      isSettingPageOpened = !isSettingPageOpened;
    }
    ImGui::Separator();

    ImGui::EndMainMenuBar();
  }
}

void MainWindow::CreateMetricsPanel(ImVec2 pos, ImVec2 size, float padding) {

  ImVec2 _padding = {padding, padding};

  // ImVec2 originalPos = ImGui::GetCursorScreenPos();
  {
    ImGui::SetCursorScreenPos(pos);
    if (ImGui::BeginChild("metrics", size)) {

      ImGui::GetWindowDrawList()->AddRectFilled(pos, pos + size, 0xAAAAAAAA);

      ImGui::SetCursorScreenPos(pos + _padding);
      if (ImGui::BeginChild("metrics content", size - _padding * 2)) {

        ImGui::Text("FPS: %.1f", _frameRate);
      }
      ImGui::EndChild();
    }
    ImGui::EndChild();
  }
  // ImGui::SetCursorScreenPos(originalPos);
}

void MainWindow::CreateMainView() {

  if (ImGui::Begin("main view", nullptr, flag)) {

    ImGui::SetWindowPos({windowPos.x, windowPos.y + menubarOffsetY});
    ImGui::SetWindowSize({mainWidth, mainHeight});

    if (ImGui::BeginTabBar("main view tab bar")) {

      for (int i = 0; i < panels.size(); i++) {
        if (ImGui::BeginTabItem(panels[i]->name().c_str())) {

          currentPanel = i;
          static const ImVec2 metricsWindowSize = {128, 128};
          ImVec2 pos = ImGui::GetCursorScreenPos();
          pos.x += ImGui::GetContentRegionAvail().x - metricsWindowSize.x;

          const ImVec2 _size = ImGui::GetContentRegionAvail();
          panels[i]->onResize(_size.x, _size.y);
          panels[i]->render();

          CreateMetricsPanel(pos, metricsWindowSize);

          ImGui::EndTabItem();
        }
      }

      ImGui::EndTabBar();
    }

    ImGui::End();
  }
}

void MainWindow::CreateParameterizationPanel() {

  if (ImGui::Begin("parameterization", nullptr, flag)) {
    ImGui::SetWindowPos({windowPos.x + mainWidth, windowPos.y + menubarOffsetY});
    ImGui::SetWindowSize({windowWidth - mainWidth, windowWidth - mainWidth});

    panels[currentPanel]->renderParameterization();

    ImGui::End();
  }
}

void MainWindow::CreateControlPanel() {

  if (ImGui::Begin("controls", nullptr, flag)) {

    float ww = windowWidth - mainWidth;
    ImGui::SetWindowPos({windowPos.x + mainWidth, windowPos.y + menubarOffsetY + ww});
    ImGui::SetWindowSize({ww, mainHeight - ww});
    {
      // title
      ImGui::NewLine();
      const char *text = "Controls";
      ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::CalcTextSize(text).x) * 0.5f);
      ImGui::Text(text);
      ImGui::NewLine();

      panels[currentPanel]->controls();

      ImGui::NewLine();
    }

    ImGui::End();
  }
}

void MainWindow::CreateSettingPage() {

  if (isSettingPageOpened && ImGui::Begin("Settings Page", &isSettingPageOpened,
                                          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {

    ImGui::SliderFloat("sliderFloat", &sliderFloat, 0.001f, 0.01f, "%.3f");

    ImGui::End();
  }
}
