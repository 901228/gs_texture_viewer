#ifndef MODEL_HPP
#define MODEL_HPP
#pragma once

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include <IconsFont/IconsLucide.h>

#include "../cache.hpp"
#include "../gl/program.hpp"
#include "../texture/texture.hpp"
#include "../texture/texture_editor.hpp"
#include "geodesic_splines.hpp"
#include "hit_test.hpp"
#include "mesh.hpp"
#include "solve_uv.hpp"

class FrameBufferHelper;
class Camera;

typedef std::pair<std::pair<float, float>, std::pair<float, float>> TextureLine;

namespace Light {
static constexpr const char *icon = ICON_LC_LIGHTBULB;
}

class Model : public GeodesicSplines::Implicit, public TextureEditor::TextureEditableModel {
public:
  explicit Model();
  explicit Model(const char *path);
  virtual ~Model();

  static constexpr const char *icon = ICON_LC_BOX;

protected:
  MyMesh _mesh;

public:
  MyMesh &mesh() { return _mesh; }

protected:
  std::unique_ptr<Program> _renderingProgram;
  unsigned int _vertexArrayObject = 0;
  unsigned int *_vertexBufferObject = nullptr;
  int _elementAmount = 0;
  std::vector<glm::vec3> _vertices;
  std::vector<glm::vec3> _normals;

protected:
  // model
  bool loadModel(const char *path);
  virtual void initMesh();

public:
  [[nodiscard]] inline const std::vector<glm::vec3> &vertices() const { return _vertices; }
  [[nodiscard]] inline const std::vector<glm::vec3> &normals() const { return _normals; }

public:
  [[nodiscard]] size_t n_faces() const;
  [[nodiscard]] size_t n_vertices() const;

public:
  void use();
  static void unUse();

public:
  void setupUniforms(const Camera &camera, bool isWire = false, bool isRenderTextureCoords = false,
                     bool isRenderTexture = false, int currentTextureId = -1,
                     const std::vector<std::unique_ptr<ImageTexture>> &textureList = {},
                     float textureRadius = 0, const glm::vec2 &textureOffset = {}, float textureTheta = 0,
                     const glm::vec3 &lightDirection = {0.0f, -1.0f, 0.0f}, float lightIntensity = 1.0f);
  virtual void render(const Camera &camera, bool renderSelectedOnly, bool isWire, bool isRenderTextureCoords,
                      bool isRenderTexture, int currentTextureId,
                      const std::vector<std::unique_ptr<ImageTexture>> &textureList, float textureRadius,
                      const glm::vec2 &textureOffset, float textureTheta, PBRTexture *pbrTexture,
                      const glm::vec3 &lightDirection, float lightIntensity);

protected:
  std::unique_ptr<std::unordered_set<unsigned int>> _selectedID;

protected:
  BVH::BVH _bvh;

public:
  inline void addSelectedID(unsigned int id) { _selectedID->insert(id); }

  [[nodiscard]] inline const std::unordered_set<unsigned int> &selectedID() const { return *_selectedID; }
  std::vector<TextureLine> getSelectedTextureLines();
  [[nodiscard]] virtual std::vector<std::pair<unsigned int, std::pair<float, float>>>
  getSelectedTextureCoords() const;

protected:
  virtual void updateTexcoordVAO();

protected:
  glm::vec3 _boxmin{std::numeric_limits<float>::max()};
  glm::vec3 _boxmax{-std::numeric_limits<float>::max()};

public:
  [[nodiscard]] glm::vec3 boxMin() const;
  [[nodiscard]] glm::vec3 boxMax() const;
  [[nodiscard]] glm::vec3 center() const;

protected:
  int _tessLevel = 4;

public:
  [[nodiscard]] inline int tessLevel() const { return _tessLevel; }
  inline void setTessLevel(int level) { _tessLevel = std::clamp(level, 1, GL_MAX_TESS_GEN_LEVEL); }

private:
  struct GlmHash {
    size_t operator()(const glm::vec3 &p) const {
      size_t seed = 0;
      Cache::hash_combine(seed, p.x);
      Cache::hash_combine(seed, p.y);
      return seed;
    }
  };

  Cache::LRUCache<glm::vec3, ClosestPointResult, GlmHash> _cache{20};
  const ClosestPointResult _closestPoint(const glm::vec3 &x);

public:
  const float eval(const glm::vec3 &x) override;
  const glm::vec3 grad(const glm::vec3 &x) override;
  const glm::vec3 project(const glm::vec3 &x) override;
  const glm::vec3 normal(const glm::vec3 &x) override;

public:
  std::optional<glm::vec3> hit(const Camera &camera, const glm::vec2 &ndcPos) const override;
  virtual bool select(const glm::vec3 &hitPoint, int radius, bool isAdd) override;
  virtual void clearSelect() override;
  virtual void solve(SolveUV::SolvingMode solvingMode,
                     std::optional<glm::vec3> hitPoint = std::nullopt) override;
  virtual void updateTextureInfo(const TextureEditor &textureEditor) override;
};

#endif // !MODEL_HPP
