#include "gs_view.hpp"

#include <glad/gl.h>

#include <cuda_gl_interop.h>

#include "../model/gs_model.hpp"
#include "../model/utils.hpp"
#include "../utils/camera/camera.hpp"
#include "../utils/logger.hpp"
#include "renderer/ellipsoid_renderer.hpp"
#include "renderer/gaussian_renderer.hpp"
#include "renderer/point_renderer.hpp"

GaussianView::GaussianView() {

  _initInterop();

  _ellipsoidRenderer = std::make_unique<EllipsoidRenderer>();
  _pointRenderer = std::make_unique<PointRenderer>();
  _gaussianRenderer = std::make_unique<GaussianRenderer>();
}

GaussianView::~GaussianView() { _releaseInterop(); }

void GaussianView::_initInterop() {

  // Create GL buffer ready for CUDA/GL interop
  glCreateBuffers(1, &imageBuffer);
  glNamedBufferStorage(imageBuffer, static_cast<long>(_width * _height * 3 * sizeof(float)), nullptr,
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
    fallback_bytes.resize(_width * _height * 3 * sizeof(float));
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

void GaussianView::_onResize(int width, int height) {
  if (_width == width && _height == height)
    return;

  _width = width;
  _height = height;

  _releaseInterop();
  _interop_failed = false; // reset state, let _initInterop re-judge
  _initInterop();
}

unsigned int GaussianView::render(RenderingMode mode, Camera &camera, int width, int height,
                                  glm::vec3 clearColor, const GaussianModel &model,
                                  const std::function<void(float *)> &render_fn) {

  _onResize(width, height);

  if (mode == RenderingMode::Points) {
    _pointRenderer->render(model.count(), model.gaussianGLData(), _width, _height, camera, clearColor);
    return _pointRenderer->getTexture();
  } else if (mode == RenderingMode::Ellipsoids) {
    _ellipsoidRenderer->render(model.count(), model.gaussianGLData(), _width, _height, camera, clearColor);
    return _ellipsoidRenderer->getTexture();
  } else if (mode == RenderingMode::Splats) {

    float *image_cuda = nullptr;
    if (!_interop_failed) {
      // Map OpenGL buffer resource for use with CUDA
      size_t bytes;
      CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
      CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&image_cuda, &bytes, imageBufferCuda));
    } else {
      image_cuda = fallbackBufferCuda;
    }

    if (!render_fn) {
      model.render(camera, _width, _height, clearColor, image_cuda);
    } else {
      render_fn(image_cuda);
    }

    if (!_interop_failed) {
      // Unmap OpenGL resource for use with OpenGL
      CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
    } else {
      CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(),
                                cudaMemcpyDeviceToHost));
      glNamedBufferSubData(imageBuffer, 0, static_cast<long>(fallback_bytes.size()), fallback_bytes.data());
    }

    // Copy image contents to framebuffer
    _gaussianRenderer->render(imageBuffer, _width, _height, clearColor, true, true);

    if (cudaPeekAtLastError() != cudaSuccess) {
      throw std::runtime_error(std::format("A CUDA error occurred during rendering:{}. Please rerun "
                                           "in Debug to find the exact line!",
                                           cudaGetErrorString(cudaGetLastError())));
    }

    return _gaussianRenderer->getTexture();
  } else {
    throw std::runtime_error("Unknown rendering mode!");
  }
}
