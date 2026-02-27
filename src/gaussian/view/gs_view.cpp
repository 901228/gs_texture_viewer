#include "gs_view.hpp"

#include <cuda_gl_interop.h>

#include <cstdio>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

#include "../model/gs_model.hpp"
#include "../model/utils.hpp"
#include "../renderer/ellipsoid_renderer.hpp"
#include "../renderer/gaussian_renderer.hpp"
#include "../renderer/point_renderer.hpp"
#include "../utils/camera/camera.hpp"
#include "../utils/logger.hpp"
#include "../utils/utils.hpp"

GaussianView::GaussianView(int render_w, int render_h, int sh_degree, bool white_bg, bool useInterop,
                           int device)
    : _render_w(render_w), _render_h(render_h), _white_bg(white_bg), _useInterop(useInterop) {

  _initInterop();
}

GaussianView::GaussianView(int render_w, int render_h, const char *plyPath, int sh_degree, bool white_bg,
                           bool useInterop, int device)
    : GaussianView(render_w, render_h, sh_degree, white_bg, useInterop, device) {

  _gsModel = std::make_unique<GaussianModel>(plyPath, sh_degree, device);

  _ellipsoidRenderer = std::make_unique<EllipsoidRenderer>();
  _pointRenderer = std::make_unique<PointRenderer>();
  _gaussianRenderer = std::make_unique<GaussianRenderer>();
}

void GaussianView::_initInterop() {

  // Create GL buffer ready for CUDA/GL interop
  glCreateBuffers(1, &imageBuffer);
  glNamedBufferStorage(imageBuffer, static_cast<long>(_render_w * _render_h * 3 * sizeof(float)), nullptr,
                       GL_DYNAMIC_STORAGE_BIT);

  if (_useInterop) {
    if (cudaPeekAtLastError() != cudaSuccess) {
      throw std::runtime_error(std::format("A CUDA error occurred in setup:{}. Please rerun in "
                                           "Debug to find the exact line!",
                                           cudaGetErrorString(cudaGetLastError())));
    }
    cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
    _useInterop &= (cudaGetLastError() == cudaSuccess);
  }

  INFO("Interop: {}", _useInterop ? "true" : "false");
  if (!_useInterop) {
    fallback_bytes.resize(_render_w * _render_h * 3 * sizeof(float));
    cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
    _interop_failed = true;
  }
}

void GaussianView::_releaseInterop() {

  if (!_interop_failed) {
    cudaGraphicsUnregisterResource(imageBufferCuda);
    imageBufferCuda = nullptr;
  } else {
    cudaFree(fallbackBufferCuda);
    fallbackBufferCuda = nullptr;
  }
  glDeleteBuffers(1, &imageBuffer);
  imageBuffer = 0;
}

GaussianView::~GaussianView() { _releaseInterop(); }

bool GaussianView::onResize(int width, int height) {
  if (_render_w == width && _render_h == height)
    return false;

  _render_w = width;
  _render_h = height;

  _releaseInterop();
  _interop_failed = false; // reset state, let _initInterop re-judge
  _initInterop();

  return true;
}

void GaussianView::render(Camera &camera) {

  glm::vec3 clearColor = _white_bg ? glm::vec3(1.0f) : glm::vec3(0.0f);

  if (currMode == RenderingMode::Points) {
    _pointRenderer->render(_gsModel->count(), _gsModel->gaussianGLData(), _render_w, _render_h, camera,
                           clearColor);
  } else if (currMode == RenderingMode::Ellipsoids) {
    _ellipsoidRenderer->render(_gsModel->count(), _gsModel->gaussianGLData(), _render_w, _render_h, camera,
                               clearColor);
  } else if (currMode == RenderingMode::Splats) {

    float *image_cuda = nullptr;
    if (!_interop_failed) {
      // Map OpenGL buffer resource for use with CUDA
      size_t bytes;
      CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
      CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&image_cuda, &bytes, imageBufferCuda));
    } else {
      image_cuda = fallbackBufferCuda;
    }

    _gsModel->render(camera, _render_w, _render_h, clearColor, image_cuda);

    if (!_interop_failed) {
      // Unmap OpenGL resource for use with OpenGL
      CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
    } else {
      CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(),
                                cudaMemcpyDeviceToHost));
      glNamedBufferSubData(imageBuffer, 0, static_cast<long>(fallback_bytes.size()), fallback_bytes.data());
    }

    // Copy image contents to framebuffer
    _gaussianRenderer->render(imageBuffer, _render_w, _render_h, clearColor, true, true);
  } else {
    throw std::runtime_error("Unknown rendering mode!");
  }

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error(std::format("A CUDA error occurred during rendering:{}. Please rerun "
                                         "in Debug to find the exact line!",
                                         cudaGetErrorString(cudaGetLastError())));
  }
}

unsigned int GaussianView::getTextureId() const {

  if (currMode == RenderingMode::Points) {
    return _pointRenderer->getTexture();
  } else if (currMode == RenderingMode::Ellipsoids) {
    return _ellipsoidRenderer->getTexture();
  } else if (currMode == RenderingMode::Splats) {
    return _gaussianRenderer->getTexture();
  } else {
    throw std::runtime_error("Unknown rendering mode!");
  }
}

void GaussianView::controls() {

  if (ImGui::Combo("Render Mode", reinterpret_cast<int *>(&currMode),
                   Utils::enumToCombo<RenderingMode>().c_str())) {

    setMode(currMode);
    DEBUG("change rendering mode to {}", Utils::name(currMode));
  }
  ImGui::NewLine();

  if (currMode == GaussianView::RenderingMode::Splats) {

    _gsModel->controls();
  }
}

GaussianModel &GaussianView::model() const { return *_gsModel; }
