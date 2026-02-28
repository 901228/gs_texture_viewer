#ifndef GAUSSIAN_VIEW_HPP
#define GAUSSIAN_VIEW_HPP
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include <glm/glm.hpp>

class EllipsoidRenderer;
class PointRenderer;
class GaussianRenderer;
class Camera;
class GaussianModel;

class GaussianView {
public:
  static GaussianView &getInstance() {
    static GaussianView instance;
    return instance;
  }
  GaussianView(GaussianView const &) = delete;
  void operator=(GaussianView const &) = delete;

private:
  GaussianView();
  ~GaussianView();

private:
  int _width = 800;
  int _height = 800;
  bool _useInterop = true;

private:
  // rendering
  unsigned int imageBuffer = 0;
  cudaGraphicsResource_t imageBufferCuda{};

  bool _interop_failed = false;
  std::vector<char> fallback_bytes{};
  float *fallbackBufferCuda = nullptr;

protected:
  std::unique_ptr<EllipsoidRenderer> _ellipsoidRenderer;
  std::unique_ptr<PointRenderer> _pointRenderer;
  std::unique_ptr<GaussianRenderer> _gaussianRenderer;

public:
  enum class RenderingMode : int { Splats, Points, Ellipsoids };

private:
  void _initInterop();
  void _releaseInterop();

  void _onResize(int width, int height);

public:
  unsigned int render(RenderingMode mode, Camera &camera, int width, int height, glm::vec3 clearColor,
                      const GaussianModel &model, const std::function<void(float *)> &render_fn = nullptr);
};

#endif // !GAUSSIAN_VIEW_HPP
