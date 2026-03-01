#ifndef TEXTURE_GS_MODEL_HPP
#define TEXTURE_GS_MODEL_HPP
#pragma once

#include "../utils/mesh/model.hpp"
#include "gs_model.hpp"

#include "rasterizer/texture_rasterizer.hpp"

class TextureGaussianModel : public GaussianModel, public Model {
public:
  TextureGaussianModel(const char *geometryPlyPath, const char *appearancePlyPath, int sh_degree,
                       int device = 0);
  ~TextureGaussianModel() override;

  void render(const Camera &camera, const int &width, const int &height, const glm::vec3 &clearColor,
              float *image_cuda, cudaTextureObject_t texId,
              const CudaRasterizer::TextureOption &textureOption = {});
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
  void selectRadius(int id, int radius, bool isAdd) override;
  void clearSelect() override;
  void calculateParameterization(SolveUV::SolvingMode solvingMode, float angle) override;
  [[nodiscard]] std::vector<std::pair<unsigned int, std::pair<float, float>>>
  getSelectedTextureCoords() const override;

private:
  // model
  glm::vec2 *updateTexcoordVAO(bool returnData) override;

private:
  float _threshold = 0.002f;

private:
  // for CUDA
  void initModelForCuda();

private:
  // screen mask

  // input
  float *_model_view_cuda = nullptr;
  float *_model_proj_cuda = nullptr;
  float *_model_position_cuda = nullptr;
  float *_model_texCoords_cuda = nullptr;
  cudaTextureObject_t *_model_sl_cuda = nullptr;
  std::uint8_t *_model_face_mask_cuda = nullptr;

  // output
  CudaRasterizer::PixelMask *_mask_cuda = nullptr;
  CudaRasterizer::RenderingMode mode = CudaRasterizer::RenderingMode::Texture;

public:
  [[nodiscard]] int count() const override;
};

#endif // !TEXTURE_GS_MODEL_HPP
