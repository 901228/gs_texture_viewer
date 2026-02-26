#include "texture_gs_view.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <texture_types.h>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

#include "../model/utils.hpp"
#include "../utils/camera/camera.hpp"
#include "renderer/gaussian_renderer.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

TextureGSView::TextureGSView(int render_w, int render_h, const char *geometryPlyPath,
                             const char *appearancePlyPath, int sh_degree, bool white_bg, bool useInterop,
                             int device)
    : GaussianView(render_w, render_h, sh_degree, white_bg, useInterop, device),
      _modelFBO(std::make_unique<FrameBufferHelper>(false, true)),
      _textureGSModel(
          std::make_unique<TextureGaussianModel>(geometryPlyPath, appearancePlyPath, sh_degree, device)) {

  _gaussianRenderer = std::make_unique<GaussianRenderer>();
}

TextureGSView::~TextureGSView() = default;

bool TextureGSView::onResize(int width, int height) {
  if (!GaussianView::onResize(width, height))
    return false;

  _modelFBO->onResize(width, height);

  return true;
}

void TextureGSView::controls() {

  if (ImGui::Combo("Render Mode", reinterpret_cast<int *>(&currMode), "Splats\0Mesh\0")) {

    setMode(currMode);
    DEBUG("change rendering mode to {}", TextureGSView::getModeName(currMode));
  }
  ImGui::NewLine();

  if (currMode == TextureGSView::RenderingMode::Splats) {

    _textureGSModel->controls();
  }
}

void TextureGSView::renderGS(const Camera &camera, cudaTextureObject_t texId, float textureRadius,
                             glm::vec2 textureOffset, float textureTheta) {

  glm::vec3 clearColor = _white_bg ? glm::vec3(1.0f) : glm::vec3(0.0f);

  float *image_cuda = nullptr;
  if (!_interop_failed) {
    // Map OpenGL buffer resource for use with CUDA
    size_t bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&image_cuda, &bytes, imageBufferCuda));
  } else {
    image_cuda = fallbackBufferCuda;
  }

  _textureGSModel->render(camera, _render_w, _render_h, clearColor, image_cuda, texId, textureRadius,
                          textureOffset, textureTheta);

  if (!_interop_failed) {
    // Unmap OpenGL resource for use with OpenGL
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
  } else {
    CUDA_SAFE_CALL(
        cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
    glNamedBufferSubData(imageBuffer, 0, static_cast<long>(fallback_bytes.size()), fallback_bytes.data());
  }

  // Copy image contents to framebuffer
  _gaussianRenderer->render(imageBuffer, _render_w, _render_h, clearColor, true, true);

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error(std::format("A CUDA error occurred during rendering:{}. Please rerun "
                                         "in Debug to find the exact line!",
                                         cudaGetErrorString(cudaGetLastError())));
  }
}

void TextureGSView::renderMesh(const Camera &camera, bool isSelect, bool renderSelectedOnly, bool isWire,
                               bool isRenderTextureCoords, bool isRenderTexture, int currentTextureId,
                               const std::vector<std::unique_ptr<ImageTexture>> &textureList,
                               float textureRadius, const glm::vec2 &textureOffset, float textureTheta) {

  // render mesh model
  {
    _modelFBO->bindDraw();
    {
      float backgroundColor = 1.0f;
      const GLfloat background[] = {backgroundColor, backgroundColor, backgroundColor, 1.0f};
      // 0.0f};
      const GLfloat one = 1.0f;

      glClearColor(background[0], background[1], background[2], background[3]);
      // NOLINTNEXTLINE(hicpp-signed-bitwise)
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glClearBufferfv(GL_COLOR, 0, background);
      glClearBufferfv(GL_DEPTH, 0, &one);

      const GLboolean cullingWasEnabled = glIsEnabled(GL_CULL_FACE);
      glEnable(GL_CULL_FACE);
      glCullFace(GL_BACK);

      mesh().render(camera, false, renderSelectedOnly, isWire, isRenderTextureCoords, isRenderTexture,
                    currentTextureId, textureList, textureRadius, textureOffset, textureTheta);

      if (!cullingWasEnabled) {
        glDisable(GL_CULL_FACE);
      }
    }
    FrameBufferHelper::unbindDraw();
  }
}

unsigned int TextureGSView::getTextureId() const {

  if (currMode == RenderingMode::Mesh) {
    return _modelFBO->getTextureId();
  } else if (currMode == RenderingMode::Splats) {
    return _gaussianRenderer->getTexture();
  } else {
    throw std::runtime_error("Unknown rendering mode!");
  }
}

GaussianModel &TextureGSView::model() const { return *_textureGSModel; }

Model &TextureGSView::mesh() const { return *_textureGSModel; }
