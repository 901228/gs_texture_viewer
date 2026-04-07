#ifndef TEXTURE_GS_MODEL_HPP
#define TEXTURE_GS_MODEL_HPP
#pragma once

#include "gs_model.hpp"
#include "utils/mesh/model.hpp"
#include "utils/texture/texture_editor.hpp"

#include "rasterizer/defines.hpp"

class TextureGaussianModel : public GaussianModel, public Model {
public:
  TextureGaussianModel(const char *geometryPlyPath, const char *appearancePlyPath, int sh_degree,
                       int device = 0);
  ~TextureGaussianModel() override;

  void render(const Camera &camera, const int &width, const int &height, const glm::vec3 &clearColor,
              float *image_cuda, TextureEditor &textureEditor,
              CudaRasterizer::MaskCullingMode maskCullingMode, CudaRasterizer::Light light);
  void controls() override;

private:
  size_t pixels = 1;

private:
  // load geometry & appearance ply
  void _loadPly(const char *geometryPlyPath, const char *appearancePlyPath);

  struct AppearancePoint {
    const glm::vec3 pos;
    const glm::vec3 berycentric;
    const unsigned int faceId;
    glm::vec2 uv;

    AppearancePoint(const glm::vec3 &pos, const glm::vec3 &berycentric, unsigned int faceId,
                    const glm::vec2 &uv = {})
        : pos(pos), berycentric(berycentric), faceId(faceId), uv(uv) {}
  };

  // appearance gaussian data
  int _gsCountA = 0;
  std::vector<AppearancePoint> _appearancePoints{};

public:
  // model
  void updateData();
  bool select(const glm::vec3 &hitPoint, int radius, bool isAdd) override;
  void clearSelect() override;
  void solve(SolveUV::SolvingMode solvingMode, std::optional<glm::vec3> hitPoint = std::nullopt) override;

  [[nodiscard]] std::vector<std::pair<unsigned int, std::pair<float, float>>>
  getSelectedTextureCoords() const override;

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
  unsigned int *_appearance_face_idx_cuda = nullptr;
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

#endif // !TEXTURE_GS_MODEL_HPP
