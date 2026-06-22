#ifndef TEXTURE_GS_MODEL_MESH_HPP
#define TEXTURE_GS_MODEL_MESH_HPP
#pragma once

#include "gs_model.hpp"
#include "utils/mesh/model.hpp"
#include "utils/texture/texture_editor.hpp"

#include "rasterizer/defines.hpp"

class TextureGaussianModelMesh : public GaussianModel, public Model {
public:
  TextureGaussianModelMesh(const char *plyPath, const char *meshPath, int sh_degree, int device = 0);
  ~TextureGaussianModelMesh() override;

  void render(const Camera &camera, const int &width, const int &height, const glm::vec3 &clearColor,
              float *image_cuda, TextureEditor &textureEditor,
              CudaRasterizer::MaskCullingMode maskCullingMode, CudaRasterizer::Light light);
  void controls() override;

private:
  size_t pixels = 1;

public:
  // model
  void updateData();
  bool select(const glm::vec3 &hitPoint, int radius, bool isAdd) override;
  void clearSelect() override;

private:
  // model
  void updateTexcoordVAO() override;

private:
  float _threshold1 = 0.000f;
  float _threshold2 = 0.000f;
  float _threshold3 = 0.010f;
  float _threshold4 = 0.000f;

private:
  // for CUDA
  std::vector<glm::vec3> _normal;
  void initMesh() override;

private:
  // screen mask

  // input
  float *_view_cuda = nullptr;
  float *_proj_cuda = nullptr;
  float *_model_position_cuda = nullptr;
  float *_model_normal_cuda = nullptr;
  float *_model_texCoords_cuda = nullptr;
  float *_model_tangent_cuda = nullptr;
  float *_model_bitangent_cuda = nullptr;
  cudaTextureObject_t *_model_basecolor_map_cuda = nullptr;
  cudaTextureObject_t *_model_normal_map_cuda = nullptr;
  cudaTextureObject_t *_model_height_map_cuda = nullptr;
  cudaTextureObject_t *_model_roughness_map_cuda = nullptr;
  cudaTextureObject_t *_model_mask_filter_cuda = nullptr;
  unsigned int *_selected_face_idx_cuda = nullptr;

  // output
  CudaRasterizer::PixelMask *_mask_cuda = nullptr;
  CudaRasterizer::RenderingMode _renderingMode = CudaRasterizer::RenderingMode::Color;

public:
  void updateTextureInfo(const TextureEditor &textureEditor) override;

public:
  using GaussianModel::center;
  [[nodiscard]] int count() const override;
};

#endif // !TEXTURE_GS_MODEL_MESH_HPP
