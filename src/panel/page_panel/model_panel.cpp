#include "model_panel.hpp"

#include <glad/gl.h>

#include "main_window.hpp"
#include "utils/camera/trackball_camera.hpp"
#include "utils/imgui/opengl.hpp"
#include "utils/mesh/model.hpp"
#include "utils/texture/texture_editor.hpp"
#include "utils/utils.hpp"

ModelPanel::ModelPanel() : model(nullptr), camera(nullptr), _textureEditor(nullptr) {}

ModelPanel::~ModelPanel() { detach(); }

void ModelPanel::_attach() {

  model = std::make_unique<Model>((char *)(PROJECT_DIR "/assets/models/armadillo.obj"));
  camera = std::make_unique<TrackballCamera>(-6.0f);
  _textureEditor = std::make_unique<TextureEditor>(*model);
}

void ModelPanel::_detach() {}

void ModelPanel::_onResize(float width, float height) {

  // projection matrix
  camera->onResize(width, height);
}

void ModelPanel::_render() {

  ImVec2 pos = ImGui::GetCursorScreenPos();

  if (ImGui::BeginOpenGL("OpenGL", {_width, _height}, false, MainWindow::flag)) {

    float backgroundColor = 1.0f;
    static const GLfloat background[] = {backgroundColor, backgroundColor, backgroundColor, 1.0f};
    static const GLfloat one = 1.0f;

    glClearColor(backgroundColor, backgroundColor, backgroundColor, 1);
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearBufferfv(GL_COLOR, 0, background);
    glClearBufferfv(GL_DEPTH, 0, &one);

    model->render(*camera, false, false, wire, _renderingMode == RenderingMode::TextureCoords,
                  _renderingMode == RenderingMode::Texture, _textureEditor->selected(),
                  _textureEditor->textureList(), _textureEditor->scale(), _textureEditor->offset(),
                  _textureEditor->theta());

    _textureEditor->handleBrushInput(*camera, _width, _height);

    camera->handleInput(pos);
  }
  ImGui::EndOpenGL();
}

void ModelPanel::_renderParameterization() {

  const glm::vec2 contentSize = {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *drawList = ImGui::GetWindowDrawList();

  // draw image
  _textureEditor->renderImage();

  // draw texture coords
  std::vector<TextureLine> lines = model->getSelectedTextureLines();
  if (!lines.empty()) {

    // draw texture coords
    for (const TextureLine &line : lines) {
      float x0 = pos.x + line.first.first * contentSize.x;
      float y0 = pos.y + line.first.second * contentSize.y;
      float x1 = pos.x + line.second.first * contentSize.x;
      float y1 = pos.y + line.second.second * contentSize.y;
      drawList->AddLine({x0, y0}, {x1, y1}, 0xFF000000, // black
                        1);
    }
  }

  // handle parameterization input
  _textureEditor->handleTextureInput();
}

void ModelPanel::_controls() {

  if (ImGui::BeginTabBar("model panel control tab bar")) {

    if (ImGui::BeginTabItem("options")) {

      ImGui::SeparatorText("Render Option");
      {
        ImGui::Checkbox("wire", &wire);
        ImGui::Combo("Rendering Mode", reinterpret_cast<int *>(&_renderingMode),
                     Utils::enumToImGuiCombo<RenderingMode>().c_str());
      }
      ImGui::NewLine();

      // TODO: add render mode (texture coords, mesh, texture)

      camera->controls({});

      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("textures")) {

      _textureEditor->controls();

      ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
  }
}
