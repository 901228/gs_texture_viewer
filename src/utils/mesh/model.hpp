#ifndef MODEL_HPP
#define MODEL_HPP
#pragma once

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include <IconsFont/IconsLucide.h>

#include "../gl/program.hpp"
#include "../texture/texture.hpp"
#include "hit_test.hpp"
#include "mesh.hpp"
#include "solve_uv.hpp"

class FrameBufferHelper;
class Camera;

typedef std::pair<std::pair<float, float>, std::pair<float, float>> TextureLine;

class TextureEditor;

namespace Light {
static constexpr const char *icon = ICON_LC_LIGHTBULB;
}

class Model {
public:
  explicit Model();
  explicit Model(const char *path);
  virtual ~Model();

  static constexpr const char *icon = ICON_LC_BOX;

protected:
  MyMesh _mesh;

protected:
  std::unique_ptr<Program> _renderingProgram;
  unsigned int _vertexArrayObject = 0;
  unsigned int *_vertexBufferObject = nullptr;
  int _elementAmount = 0;
  std::vector<glm::vec3> _vertices;

protected:
  // model
  bool loadModel(const char *path);
  virtual void initMesh();
  [[nodiscard]] inline const std::vector<glm::vec3> &vertices() const { return _vertices; }

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

public:
  BVH::BVH _bvh;
  [[nodiscard]] HitResult select(const Camera &camera, float width, float height,
                                 const glm::vec2 &mousePos) const;

  inline void addSelectedID(unsigned int id) { _selectedID->insert(id); }
  virtual void clearSelect();
  virtual bool selectRadius(int id, int radius, bool isAdd);

  virtual void calculateParameterization(SolveUV::SolvingMode solvingMode);

  [[nodiscard]] inline const std::unordered_set<unsigned int> &selectedID() const { return *_selectedID; }
  std::vector<TextureLine> getSelectedTextureLines();
  [[nodiscard]] virtual std::vector<std::pair<unsigned int, std::pair<float, float>>>
  getSelectedTextureCoords() const;

protected:
  virtual void updateTexcoordVAO();

public:
  virtual void updateTexId(TextureEditor &textureEditor);

protected:
  glm::vec3 _boxmin{std::numeric_limits<float>::max()};
  glm::vec3 _boxmax{-std::numeric_limits<float>::max()};

public:
  [[nodiscard]] glm::vec3 center() const;

protected:
  int _tessLevel = 4;

public:
  [[nodiscard]] inline int tessLevel() const { return _tessLevel; }
  inline void setTessLevel(int level) { _tessLevel = std::clamp(level, 1, GL_MAX_TESS_GEN_LEVEL); }
};

#endif // !MODEL_HPP
