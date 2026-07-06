#ifndef GLTF_MODEL_HPP
#define GLTF_MODEL_HPP
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include <IconsFont/IconsLucide.h>

#include "../light.hpp"
#include "../texture/texture_editor.hpp"
#include "hit_test.hpp"
#include "mesh.hpp"
#include "solve_uv.hpp"

class Program;
class Camera;
class ImageTexture;
class PBRTexture;

struct aiScene;
struct aiNode;
struct aiMesh;

// A line in texture space, reused by the parameterization panel. Mirrors the typedef in model.hpp.
typedef std::pair<std::pair<float, float>, std::pair<float, float>> TextureLine;

// Loads a .glb scene through assimp and renders every sub-mesh with its own PBR material
// under directional lighting. Implements the texture-editor interface so the user can paint
// a decal texture, but only onto a single "active" sub-mesh at a time (the one last clicked).
class GltfModel : public TextureEditor::TextureEditableModel {
public:
  static constexpr const char *icon = ICON_LC_BOX;

  explicit GltfModel(const char *path);
  ~GltfModel();

  void render(const Camera &camera, bool renderSelectedOnly, bool isWire, bool isRenderTextureCoords,
              bool isRenderTexture, int currentTextureId,
              const std::vector<std::unique_ptr<ImageTexture>> &textureList, float textureRadius,
              const glm::vec2 &textureOffset, float textureTheta, PBRTexture *pbrTexture,
              const std::vector<Light> &lights, bool flipNormals, bool decalNormalOnly);

  [[nodiscard]] glm::vec3 boxMin() const { return _boxmin; }
  [[nodiscard]] glm::vec3 boxMax() const { return _boxmax; }
  [[nodiscard]] glm::vec3 center() const;

  // texture-coordinate lines of the active sub-mesh, for the parameterization panel.
  std::vector<TextureLine> getSelectedTextureLines();

  // TextureEditor::TextureEditableModel
  std::optional<glm::vec3> hit(const Camera &camera, const glm::vec2 &ndcPos) const override;
  bool select(const glm::vec3 &hitPoint, int radius, bool isAdd) override;
  void clearSelect() override;
  void solve(SolveUV::SolvingMode solvingMode, std::optional<glm::vec3> hitPoint = std::nullopt) override;
  void updateTextureInfo(const TextureEditor &editor) override;

private:
  // PBR material resolved from an aiMaterial. Missing maps fall back to the scalar/vector factors.
  struct Material {
    std::unique_ptr<ImageTexture> baseColor;
    std::unique_ptr<ImageTexture> metallicRoughness; // glTF packing: G = roughness, B = metallic
    std::unique_ptr<ImageTexture> normal;
    std::unique_ptr<ImageTexture> emissive;
    std::unique_ptr<ImageTexture> occlusion;

    glm::vec4 baseColorFactor{1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    glm::vec3 emissiveFactor{0.0f};
  };

  // One aiMesh: geometry (OpenMesh for editing + GL buffers for drawing) plus its material.
  struct SubMesh {
    std::string name;

    MyMesh mesh;   // shared-vertex topology, used by BVH / brush select / UV solvers
    BVH::BVH bvh;  // built from `mesh`, face indices match mesh.face_handle(i)

    // per-vertex material data, indexed by MyMesh vertex idx (immutable after load)
    std::vector<glm::vec2> uv0;
    std::vector<glm::vec3> matTangent;
    std::vector<glm::vec3> matBitangent;

    Material material;

    // GL: vertices are expanded to face*3, in mesh face order (see initSubMeshGL).
    unsigned int vao = 0;
    unsigned int vbo[9] = {0}; // pos,normal,uv0,uvDecal,sl,matTan,matBitan,decalTan,decalBitan
    int elementCount = 0;

    std::unordered_set<unsigned int> selectedID;

    glm::vec3 boxmin{std::numeric_limits<float>::max()};
    glm::vec3 boxmax{-std::numeric_limits<float>::max()};
  };

  std::vector<SubMesh> _subMeshes;
  int _activeSubMesh = -1;

  std::unique_ptr<Program> _program;
  unsigned int _whiteTexture = 0; // 1x1 white fallback for missing material maps

  glm::vec3 _boxmin{std::numeric_limits<float>::max()};
  glm::vec3 _boxmax{-std::numeric_limits<float>::max()};

private:
  void load(const char *path);
  // Walk the scene graph, accumulating node transforms, and bake each node's world
  // transform into the meshes it references (glTF places meshes via the node hierarchy).
  void processNode(const aiNode *node, const glm::mat4 &parentTransform, const aiScene *scene,
                   const std::string &modelDir);
  void buildSubMesh(const aiScene *scene, const aiMesh *aimesh, const glm::mat4 &transform,
                    const std::string &modelDir);
  void initSubMeshGL(SubMesh &sm);
  void renderSubMesh(SubMesh &sm, bool isActive, int currentTextureId, PBRTexture *pbrTexture,
                     bool isRenderTextureCoords);
  // re-upload the decal UV / tangent buffers after a solve, and the selection marker buffer.
  void updateDecalBuffers(SubMesh &sm);
  void updateSelectBuffer(SubMesh &sm);
};

#endif // !GLTF_MODEL_HPP
