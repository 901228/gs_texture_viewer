#ifndef GS_VIEW_HPP
#define GS_VIEW_HPP
#pragma once

#include <memory>
#include <string>

#include "../model/gs_gl_data.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>

class Camera;
class EllipsoidRenderer;
class PointRenderer;
class GaussianRenderer;
class GaussianModel;

class GaussianView {
public:
  /**
   * Constructor
   *
   * @param render_w rendering width
   * @param render_h rendering height
   * @param sh_degree spherical harmonics degree
   * @param white_bg if true, render with white background, otherwise render with black background
   * @param useInterop if true, use CUDA/OpenGL interop, which will share the memory between CUDA and OpenGL
   * @param device CUDA device index
   */
  GaussianView(int render_w, int render_h, int sh_degree, bool white_bg = false, bool useInterop = true,
               int device = 0);

  GaussianView(int render_w, int render_h, const char *plyPath, int sh_degree, bool white_bg = false,
               bool useInterop = true, int device = 0);
  ~GaussianView();

  void render(Camera &camera);
  virtual bool onResize(int width, int height);
  virtual void controls();
  [[nodiscard]] virtual GaussianModel &model() const;
  [[nodiscard]] virtual unsigned int getTextureId() const;

protected:
  void _initInterop();
  void _releaseInterop();

protected:
  int _render_w;
  int _render_h;
  bool _white_bg;
  bool _useInterop;

protected:
  // gaussian model data
  std::unique_ptr<GaussianModel> _gsModel;

protected:
  // rendering
  GLuint imageBuffer = 0;
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
  inline void setMode(RenderingMode mode) { currMode = mode; }
  static inline std::string getModeName(RenderingMode mode) {
    switch (mode) {
    case RenderingMode::Splats:
      return "Splats";
    case RenderingMode::Points:
      return "InitialPoints";
    case RenderingMode::Ellipsoids:
      return "Ellipsoids";
    default:
      return "Unknown";
    }
  }

protected:
  RenderingMode currMode = RenderingMode::Splats;
};

#endif // !GS_VIEW_HPP
