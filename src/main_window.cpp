#include "main_window.hpp"

#include <glad/gl.h>

#include <ImGui/backends/imgui_impl_glfw.h>
#include <ImGui/backends/imgui_impl_opengl3.h>

#include <ImGui/imgui.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <nfd.h>
#include <stb/stb_image.h>

#include <IconsFont/IconsLucide.h>

#include "panel/page_panel/gaussian_panel.hpp"
#include "panel/page_panel/model_panel.hpp"
#include "panel/page_panel/texture_gs_panel.hpp"
#include "utils/imgui/icon.hpp"
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

    // Font
    {
      static const float baseFontSize = 13.0f; // 13.0f is the size of the default font.
      static const float iconFontSize =
          baseFontSize * 2.0f /
          3.0f; // Icon fonts need to have their sizes reduced by 2.0f/3.0f in order to align correctly
      {
        ImFontConfig icons_config;
        icons_config.SizePixels = baseFontSize;
        icons_config.PixelSnapH = false;
        io.Fonts->AddFontDefaultVector(&icons_config);
      }

      static const ImWchar lucide_icons_ranges[] = {ICON_MIN_LC, ICON_MAX_16_LC, 0};
      const std::string lucide_ttf = Utils::Path::getAssetsPath("fonts/" FONT_ICON_FILE_NAME_LC);

      // merge Lucide icon font
      {
        ImFontConfig icons_config;
        icons_config.MergeMode = true;
        icons_config.PixelSnapH = false;
        icons_config.GlyphMinAdvanceX = iconFontSize;
        // icons_config.GlyphOffset = {0, (baseFontSize - iconFontSize) / 2.0f}; // (1 - 2/3) / 2 = 1/6
        io.Fonts->AddFontFromFileTTF(lucide_ttf.c_str(), iconFontSize, &icons_config, lucide_icons_ranges);
      }

      // bigger Lucide icon font for render icon only text
      {
        static const float biggerIconFontSize = 16.0f;
        ImFontConfig icons_config;
        icons_config.MergeMode = false;
        icons_config.GlyphMinAdvanceX = biggerIconFontSize;
        ImGui::iconOnlyFont = io.Fonts->AddFontFromFileTTF(lucide_ttf.c_str(), biggerIconFontSize,
                                                           &icons_config, lucide_icons_ranges);
      }
    }

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

void MainWindow::CreateMetricsPanel(ImVec2 rightTopPos) {

  // ImVec2 originalPos = ImGui::GetCursorScreenPos();
  {
    ImVec2 panelSize{};
    const ImVec2 framePadding = ImGui::GetStyle().FramePadding;
    if (_metricsCollapsed) {
      ImGui::PushIconFont();
      panelSize = ImGui::GetButtonSize(ICON_LC_INFO);
      ImGui::PopIconFont();
    } else {
      ImGui::PushIconFont();
      ImVec2 iconButtonSize = ImGui::GetButtonSize(ICON_LC_PANEL_RIGHT_CLOSE);
      ImGui::PopIconFont();

      panelSize.x += iconButtonSize.x + _metricsContentSize.x + framePadding.x * 2;
      panelSize.y = std::max(iconButtonSize.y, _metricsContentSize.y + framePadding.y * 2);
    }

    auto metricsContent = [this]() { ImGui::Text("FPS: %.1f", _frameRate); };

    ImGui::SetCursorScreenPos({rightTopPos.x - panelSize.x, rightTopPos.y});
    if (ImGui::BeginChild("metrics", panelSize)) {

      ImGui::PushIconFont();
      if (ImGui::Button(_metricsCollapsed ? ICON_LC_INFO : ICON_LC_PANEL_RIGHT_CLOSE)) {
        _metricsCollapsed = !_metricsCollapsed;
      }
      ImGui::PopIconFont();

      if (_metricsCollapsed && ImGui::IsItemHovered() && ImGui::BeginTooltip()) {

        metricsContent();

        ImGui::EndTooltip();
      } else if (!_metricsCollapsed) {

        ImGui::SetCursorScreenPos(
            {rightTopPos.x - _metricsContentSize.x - framePadding.x, rightTopPos.y + framePadding.y});
        if (ImGui::BeginChild("metrics inner", _metricsContentSize)) {

          ImGui::BeginGroup();
          metricsContent();
          ImGui::EndGroup();

          _metricsContentSize = ImGui::GetItemRectSize();

          ImGui::EndChild();
        }
      }

      ImGui::EndChild();
    }
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
          const ImVec2 pos = ImGui::GetCursorScreenPos();
          const ImVec2 rightTopPos{pos.x + ImGui::GetContentRegionAvail().x, pos.y};

          const ImVec2 _size = ImGui::GetContentRegionAvail();
          panels[i]->onResize(_size.x, _size.y);
          panels[i]->render();

          CreateMetricsPanel(rightTopPos);

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
      panels[currentPanel]->controls();
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
