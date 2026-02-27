#ifndef TEXTURE_GS_VIEW_HPP
#define TEXTURE_GS_VIEW_HPP
#pragma once

#include <memory>

#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "../model/texture_gs_model.hpp"
#include "gs_view.hpp"

class GaussianRenderer;
class Camera;

class TextureGSView : public GaussianView {
public:
  /**
   * Constructor
   *
   * @param render_w rendering width
   * @param render_h rendering height
   * @param geometryPlyPath geometry ply path
   * @param appearancePlyPath appearance ply path
   * @param sh_degree spherical harmonics degree
   * @param white_bg if true, render with white background, otherwise render with black background
   * @param useInterop if true, use CUDA/OpenGL interop, which will share the memory between CUDA and OpenGL
   * @param device CUDA device index
   */
  TextureGSView(int render_w, int render_h, const char *geometryPlyPath, const char *appearancePlyPath,
                int sh_degree, bool white_bg = false, bool useInterop = true, int device = 0);
  ~TextureGSView();

public:
  bool onResize(int width, int height) override;
  void controls() override;

public:
  void renderGS(const Camera &camera, cudaTextureObject_t texId, float textureRadius = 1,
                glm::vec2 textureOffset = {}, float textureTheta = 0);
  void renderMesh(const Camera &camera, bool isSelect = false, bool renderSelectedOnly = false,
                  bool isWire = false, bool isRenderTextureCoords = false, bool isRenderTexture = false,
                  int currentTextureId = -1,
                  const std::vector<std::unique_ptr<ImageTexture>> &textureList = {}, float textureRadius = 0,
                  const glm::vec2 &textureOffset = {}, float textureTheta = 0);
  [[nodiscard]] unsigned int getTextureId() const override;

  [[nodiscard]] GaussianModel &model() const override;
  [[nodiscard]] Model &mesh() const;

private:
  std::unique_ptr<TextureGaussianModel> _textureGSModel;

  std::unique_ptr<FrameBufferHelper> _modelFBO;

public:
  enum class RenderingMode : int { Splats, Mesh };
  [[nodiscard]] inline RenderingMode mode() const { return currMode; }
  inline void setMode(RenderingMode mode) { currMode = mode; }

private:
  RenderingMode currMode = RenderingMode::Splats;
};

#endif // !TEXTURE_GS_VIEW_HPP
