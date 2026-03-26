#ifndef GS_MODEL_HPP
#define GS_MODEL_HPP
#pragma once

#include <memory>

#include <glm/glm.hpp>

#include "ply.hpp"
#include "utils/logger.hpp"

#define CUDA_SAFE_CALL_ALWAYS(A)                                                                             \
  A;                                                                                                         \
  cudaDeviceSynchronize();                                                                                   \
  if (cudaPeekAtLastError() != cudaSuccess)                                                                  \
    ERROR(cudaGetErrorString(cudaGetLastError()));

#if DEBUG || _DEBUG
#define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
#define CUDA_SAFE_CALL(A) A
#endif
class Camera;
class GaussianGLData;

class GaussianModel {
public:
  inline static void flipRow(glm::mat4 &mat, int row) {
    for (int c = 0; c < 4; ++c)
      mat[c][row] *= -1.0f;
  };

public:
  explicit GaussianModel(int sh_degree, int device = 0);
  GaussianModel(const char *plyPath, int sh_degree, int device = 0);
  ~GaussianModel();

protected:
  static void _initCuda(int device);
  virtual std::tuple<std::vector<Pos>, std::vector<Rot>, std::vector<Scale>, std::vector<float>>
  _loadPly(const char *plyPath);

public:
  virtual void render(const Camera &camera, const int &width, const int &height, const glm::vec3 &clearColor,
                      float *image_cuda) const;
  virtual void controls();

public:
  [[nodiscard]] glm::vec3 boxMin() const;
  [[nodiscard]] glm::vec3 boxMax() const;
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
  float *_colmap_view_cuda = nullptr;
  float *_colmap_proj_view_cuda = nullptr;
  float *_cam_pos_cuda = nullptr;

  /**
   * Convert Coordinate System from Colmap to OpenGL
   *
   * Colmap (OpenCV)      OpenGL
   *  z                    y
   *   \                   |
   *    \                  |
   *     \                 |
   *      +------x         +------x
   *      |                 \
   *      |                  \
   *      |                   \
   *      y                    z
   *
   * Flip Y & Z axis of view matrix to match OpenGL
   *
   * Only flip Y axis of projection matrix because the z of the clipped position represents the depth of the
   * point, which will be processed by division by the z of the view pos.
   */
  virtual void uploadColmapViewPorjMatrix(const Camera &camera) const;

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
