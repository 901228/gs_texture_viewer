#ifndef GS_MODEL_HPP
#define GS_MODEL_HPP
#pragma once

#include <memory>

#include <glm/glm.hpp>

class Camera;
class GaussianGLData;

class GaussianModel {
public:
  explicit GaussianModel(int sh_degree, int device = 0);
  GaussianModel(const char *plyPath, int sh_degree, int device = 0);
  ~GaussianModel();

protected:
  static void _initCuda(int device);
  void _loadPly(const char *plyPath);

public:
  virtual void render(const Camera &camera, const int &width, const int &height, const glm::vec3 &clearColor,
                      float *image_cuda);
  virtual void controls();

public:
  [[nodiscard]] glm::vec3 center() const;
  [[nodiscard]] GaussianGLData &gaussianGLData() const;
  [[nodiscard]] virtual int count() const;

protected:
  int _sh_degree;

protected:
  // gaussian parameters
  glm::vec3 _scenemin{}, _scenemax{};

  int gsCount = 0;

  // gs data buffer
  float *_pos_cuda = nullptr;
  float *_rot_cuda = nullptr;
  float *_scale_cuda = nullptr;
  float *_opacity_cuda = nullptr;
  float *_shs_cuda = nullptr;

  // rendering data buffer
  float *_view_cuda = nullptr;
  float *_proj_cuda = nullptr;
  float *_cam_pos_cuda = nullptr;
  float *_background_cuda = nullptr;
  int *_rect_cuda = nullptr;

  std::unique_ptr<GaussianGLData> _gsGLData;

  size_t _allocdGeom = 0, _allocdBinning = 0, _allocdImg = 0;
  void *_geomPtr = nullptr, *_binningPtr = nullptr, *_imgPtr = nullptr;
  std::function<char *(size_t N)> _geomBufferFunc{}, _binningBufferFunc{}, _imgBufferFunc{};

protected:
  // rendering options
  glm::vec3 _boxmin{}, _boxmax{};
  bool _fastCulling = true;
  bool _cropping = false;
  float _scalingModifier = 1.0f;
  bool _antialiasing = false;
};

#endif // !GS_MODEL_HPP
