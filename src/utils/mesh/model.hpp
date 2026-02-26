#ifndef MODEL_HPP
#define MODEL_HPP
#pragma once

#include <memory>
#include <set>
#include <vector>

#include <glm/glm.hpp>

#include "../gl/program.hpp"
#include "../texture/texture.hpp"
#include "mesh.hpp"
#include "solve_uv.hpp"

class FrameBufferHelper;
class Camera;

typedef std::pair<std::pair<float, float>, std::pair<float, float>> TextureLine;

class Model {
public:
  explicit Model();
  explicit Model(char *path);
  virtual ~Model();

protected:
  MyMesh _mesh;

private:
  std::unique_ptr<Program> _renderingProgram;
  std::unique_ptr<Program> _selectingProgram;
  unsigned int _vertexArrayObject = 0;
  unsigned int *_vertexBufferObject = nullptr;
  int _elementAmount = 0;
  std::vector<float> _vertices;

protected:
  // model
  bool loadModel(const char *path);
  void initMesh(bool toGL = true);
  [[nodiscard]] inline const std::vector<float> &vertices() const { return _vertices; }

public:
  [[nodiscard]] size_t n_faces() const;
  [[nodiscard]] size_t n_vertices() const;

public:
  const Program &use(bool isSelect = false);
  static void unUse();

public:
  static void setupUniformsCommon(const Program &program, const Camera &camera);
  void setupUniforms(const Camera &camera, bool isWire = false, bool isRenderTextureCoords = false,
                     bool isRenderTexture = false, int currentTextureId = -1,
                     const std::vector<std::unique_ptr<ImageTexture>> &textureList = {},
                     float textureRadius = 0, const glm::vec2 &textureOffset = {}, float textureTheta = 0);
  virtual void render(const Camera &camera, bool isSelect, bool renderSelectedOnly, bool isWire,
                      bool isRenderTextureCoords, bool isRenderTexture, int currentTextureId,
                      const std::vector<std::unique_ptr<ImageTexture>> &textureList, float textureRadius,
                      const glm::vec2 &textureOffset, float textureTheta);

protected:
  std::unique_ptr<std::set<unsigned int>> _selectedID;

public:
  static int getSelectedID(FrameBufferHelper &selectingFBO, int x, int y);
  inline void addSelectedID(unsigned int id) { _selectedID->insert(id); }
  virtual void clearSelect();
  virtual void selectRadius(int id, int radius, bool isAdd);

  virtual void calculateParameterization(SolveUV::SolvingMode solvingMode, float angle);

  [[nodiscard]] inline const std::set<unsigned int> &selectedID() const { return *_selectedID; }
  std::vector<TextureLine> getSelectedTextureLines();
  [[nodiscard]] virtual std::vector<std::pair<unsigned int, std::pair<float, float>>>
  getSelectedTextureCoords() const;

protected:
  virtual glm::vec2 *updateTexcoordVAO(bool returnData);
};

#endif // !MODEL_HPP
