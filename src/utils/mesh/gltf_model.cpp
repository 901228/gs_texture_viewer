#include "gltf_model.hpp"

#include <filesystem>
#include <queue>

#include <glad/gl.h>

#include <glm/gtc/type_ptr.hpp>

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "../camera/camera.hpp"
#include "../gl/program.hpp"
#include "../logger.hpp"
#include "../texture/texture.hpp"
#include "../utils.hpp"

namespace {

// assimp stores matrices row-major; glm is column-major, so transpose on conversion.
glm::mat4 toGlm(const aiMatrix4x4 &m) {
  return glm::mat4(m.a1, m.b1, m.c1, m.d1,  // column 0
                   m.a2, m.b2, m.c2, m.d2,  // column 1
                   m.a3, m.b3, m.c3, m.d3,  // column 2
                   m.a4, m.b4, m.c4, m.d4); // column 3
}

// Resolve an aiMaterial texture slot to an ImageTexture, supporting glb embedded textures
// (path "*N" -> scene->mTextures[N]) and on-disk files relative to the model directory.
std::unique_ptr<ImageTexture> loadMaterialTexture(const aiScene *scene, const aiMaterial *mat,
                                                  aiTextureType type, const std::string &modelDir,
                                                  ImageTexture::ColorType colorType) {
  if (mat->GetTextureCount(type) == 0)
    return nullptr;

  aiString path;
  if (mat->GetTexture(type, 0, &path) != AI_SUCCESS)
    return nullptr;

  // glTF textures repeat by default
  const auto wrap = TextureWrap::Mode::Repeat;

  if (const aiTexture *tex = scene->GetEmbeddedTexture(path.C_Str())) {
    if (tex->mHeight == 0) {
      // compressed (png/jpg) blob of length mWidth
      return ImageTexture::createFromMemory(path.C_Str(), reinterpret_cast<const unsigned char *>(tex->pcData),
                                            static_cast<int>(tex->mWidth), colorType, wrap, wrap);
    }
    WARN("Embedded raw (uncompressed) textures are not supported: {}", path.C_Str());
    return nullptr;
  }

  std::string filePath = (std::filesystem::path(modelDir) / path.C_Str()).string();
  return ImageTexture::create(filePath, colorType, wrap, wrap);
}

} // namespace

GltfModel::GltfModel(const char *path)
    : _program(std::make_unique<Program>(Utils::Path::getShaderPath("gltf/gltf.vert"),
                                         Utils::Path::getShaderPath("gltf/gltf.frag"), "",
                                         Utils::Path::getShaderPath("gltf/gltf.tesc"),
                                         Utils::Path::getShaderPath("gltf/gltf.tese"))) {

  // 1x1 white fallback bound to any missing material slot so the shader can always sample.
  glGenTextures(1, &_whiteTexture);
  glBindTexture(GL_TEXTURE_2D, _whiteTexture);
  const unsigned char white[4] = {255, 255, 255, 255};
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  load(path);
}

GltfModel::~GltfModel() {
  for (SubMesh &sm : _subMeshes) {
    if (sm.vbo[0] != 0)
      glDeleteBuffers(9, sm.vbo);
    if (sm.vao != 0)
      glDeleteVertexArrays(1, &sm.vao);
  }
  if (_whiteTexture != 0)
    glDeleteTextures(1, &_whiteTexture);
}

glm::vec3 GltfModel::center() const { return Utils::center(_boxmin, _boxmax); }

void GltfModel::load(const char *path) {
  Assimp::Importer importer;
  // glTF uses a top-left UV origin and textures are loaded unflipped (see createFromMemory),
  // so do NOT apply aiProcess_FlipUVs here.
  const aiScene *scene = importer.ReadFile(
      path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace |
                aiProcess_JoinIdenticalVertices);

  if (scene == nullptr || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || scene->mRootNode == nullptr) {
    ERROR("assimp failed to load {}: {}", path, importer.GetErrorString());
    return;
  }

  std::string modelDir = std::filesystem::path(path).parent_path().string();

  _subMeshes.reserve(scene->mNumMeshes);
  processNode(scene->mRootNode, glm::mat4(1.0f), scene, modelDir);

  if (_subMeshes.empty())
    ERROR("glb {} produced no renderable sub-meshes", path);
  else
    INFO("loaded glb {} with {} sub-meshes", path, _subMeshes.size());
}

void GltfModel::processNode(const aiNode *node, const glm::mat4 &parentTransform, const aiScene *scene,
                            const std::string &modelDir) {
  glm::mat4 transform = parentTransform * toGlm(node->mTransformation);

  for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
    const aiMesh *aimesh = scene->mMeshes[node->mMeshes[i]];
    if (aimesh->mNumVertices == 0 || aimesh->mNumFaces == 0)
      continue;
    buildSubMesh(scene, aimesh, transform, modelDir);
  }

  for (unsigned int i = 0; i < node->mNumChildren; ++i)
    processNode(node->mChildren[i], transform, scene, modelDir);
}

void GltfModel::buildSubMesh(const aiScene *scene, const aiMesh *aimesh, const glm::mat4 &transform,
                             const std::string &modelDir) {
  // Bake the node's world transform into the geometry; the shader's model matrix stays identity.
  const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform)));
  const glm::mat3 linear = glm::mat3(transform);

  SubMesh sm;
  sm.name = aimesh->mName.length > 0 ? aimesh->mName.C_Str() : "mesh";

  sm.mesh.request_vertex_status();
  sm.mesh.request_edge_status();
  sm.mesh.request_face_status();

  const bool hasUV = aimesh->HasTextureCoords(0);
  const bool hasTangents = aimesh->HasTangentsAndBitangents();

  sm.uv0.resize(aimesh->mNumVertices, {0.0f, 0.0f});
  sm.matTangent.resize(aimesh->mNumVertices, {0.0f, 0.0f, 0.0f});
  sm.matBitangent.resize(aimesh->mNumVertices, {0.0f, 0.0f, 0.0f});

  std::vector<MyMesh::VertexHandle> handles(aimesh->mNumVertices);
  for (unsigned int i = 0; i < aimesh->mNumVertices; ++i) {
    const aiVector3D &p = aimesh->mVertices[i];
    glm::vec3 gp = glm::vec3(transform * glm::vec4(p.x, p.y, p.z, 1.0f));
    handles[i] = sm.mesh.add_vertex({gp.x, gp.y, gp.z});

    if (aimesh->HasNormals()) {
      const aiVector3D &n = aimesh->mNormals[i];
      glm::vec3 gn = glm::normalize(normalMatrix * glm::vec3(n.x, n.y, n.z));
      sm.mesh.set_normal(handles[i], {gn.x, gn.y, gn.z});
    }
    if (hasUV)
      sm.uv0[i] = {aimesh->mTextureCoords[0][i].x, aimesh->mTextureCoords[0][i].y};
    if (hasTangents) {
      const aiVector3D &t = aimesh->mTangents[i];
      const aiVector3D &b = aimesh->mBitangents[i];
      sm.matTangent[i] = linear * glm::vec3(t.x, t.y, t.z);
      sm.matBitangent[i] = linear * glm::vec3(b.x, b.y, b.z);
    }

    // initialise the decal UV to the model's own UV0 so a decal previews sensibly
    // even before the user runs a parameterization.
    sm.mesh.set_texcoord2D(handles[i], {sm.uv0[i].x, sm.uv0[i].y});

    sm.boxmin = glm::min(sm.boxmin, gp);
    sm.boxmax = glm::max(sm.boxmax, gp);
  }

  for (unsigned int f = 0; f < aimesh->mNumFaces; ++f) {
    const aiFace &face = aimesh->mFaces[f];
    if (face.mNumIndices != 3)
      continue;
    sm.mesh.add_face(handles[face.mIndices[0]], handles[face.mIndices[1]], handles[face.mIndices[2]]);
  }

  // material
  const aiMaterial *mat = scene->mMaterials[aimesh->mMaterialIndex];
  Material &material = sm.material;

  material.baseColor =
      loadMaterialTexture(scene, mat, aiTextureType_BASE_COLOR, modelDir, ImageTexture::ColorType::Auto);
  if (!material.baseColor)
    material.baseColor =
        loadMaterialTexture(scene, mat, aiTextureType_DIFFUSE, modelDir, ImageTexture::ColorType::Auto);

  material.metallicRoughness =
      loadMaterialTexture(scene, mat, aiTextureType_METALNESS, modelDir, ImageTexture::ColorType::Auto);
  if (!material.metallicRoughness)
    material.metallicRoughness = loadMaterialTexture(scene, mat, aiTextureType_DIFFUSE_ROUGHNESS, modelDir,
                                                     ImageTexture::ColorType::Auto);

  material.normal =
      loadMaterialTexture(scene, mat, aiTextureType_NORMALS, modelDir, ImageTexture::ColorType::RGB);
  material.emissive =
      loadMaterialTexture(scene, mat, aiTextureType_EMISSIVE, modelDir, ImageTexture::ColorType::Auto);
  material.occlusion =
      loadMaterialTexture(scene, mat, aiTextureType_LIGHTMAP, modelDir, ImageTexture::ColorType::Auto);
  if (!material.occlusion)
    material.occlusion = loadMaterialTexture(scene, mat, aiTextureType_AMBIENT_OCCLUSION, modelDir,
                                             ImageTexture::ColorType::Auto);

  aiColor4D baseColor;
  if (mat->Get(AI_MATKEY_BASE_COLOR, baseColor) == AI_SUCCESS)
    material.baseColorFactor = {baseColor.r, baseColor.g, baseColor.b, baseColor.a};
  ai_real metallic;
  if (mat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == AI_SUCCESS)
    material.metallicFactor = metallic;
  ai_real roughness;
  if (mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) == AI_SUCCESS)
    material.roughnessFactor = roughness;
  aiColor3D emissive;
  if (mat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive) == AI_SUCCESS)
    material.emissiveFactor = {emissive.r, emissive.g, emissive.b};

  sm.bvh.build(sm.mesh);
  initSubMeshGL(sm);

  _boxmin = glm::min(_boxmin, sm.boxmin);
  _boxmax = glm::max(_boxmax, sm.boxmax);

  _subMeshes.push_back(std::move(sm));
}

void GltfModel::initSubMeshGL(SubMesh &sm) {
  const size_t vertexCount = sm.mesh.n_faces() * 3;

  std::vector<glm::vec3> positions, normals, matTangent, matBitangent, decalTangent, decalBitangent;
  std::vector<glm::vec2> uv0, uvDecal;
  std::vector<GLint> selectIdx;
  positions.reserve(vertexCount);
  normals.reserve(vertexCount);
  uv0.reserve(vertexCount);
  uvDecal.reserve(vertexCount);
  matTangent.reserve(vertexCount);
  matBitangent.reserve(vertexCount);
  decalTangent.reserve(vertexCount);
  decalBitangent.reserve(vertexCount);
  selectIdx.reserve(vertexCount);

  for (const MyMesh::FaceHandle &fh : sm.mesh.faces()) {
    for (const MyMesh::VertexHandle &vh : sm.mesh.fv_range(fh)) {
      positions.emplace_back(Utils::toGlm(sm.mesh.point(vh)));
      normals.emplace_back(Utils::toGlm(sm.mesh.normal(vh)));
      uv0.emplace_back(sm.uv0[vh.idx()]);
      uvDecal.emplace_back(Utils::toGlm(sm.mesh.texcoord2D(vh)));
      matTangent.emplace_back(sm.matTangent[vh.idx()]);
      matBitangent.emplace_back(sm.matBitangent[vh.idx()]);
      decalTangent.emplace_back(0.0f);
      decalBitangent.emplace_back(0.0f);
      selectIdx.emplace_back(-1);
    }
  }

  sm.elementCount = static_cast<int>(vertexCount);

  glGenVertexArrays(1, &sm.vao);
  glBindVertexArray(sm.vao);
  glGenBuffers(9, sm.vbo);

  auto uploadVec3 = [](unsigned int vbo, GLuint loc, const std::vector<glm::vec3> &data) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<long>(data.size() * sizeof(glm::vec3)), data.data(),
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(loc);
  };
  auto uploadVec2 = [](unsigned int vbo, GLuint loc, const std::vector<glm::vec2> &data) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<long>(data.size() * sizeof(glm::vec2)), data.data(),
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(loc);
  };

  uploadVec3(sm.vbo[0], 0, positions);
  uploadVec3(sm.vbo[1], 1, normals);
  uploadVec2(sm.vbo[2], 2, uv0);
  uploadVec2(sm.vbo[3], 3, uvDecal);

  glBindBuffer(GL_ARRAY_BUFFER, sm.vbo[4]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(selectIdx.size() * sizeof(GLint)), selectIdx.data(),
               GL_DYNAMIC_DRAW);
  glVertexAttribIPointer(4, 1, GL_INT, 0, nullptr);
  glEnableVertexAttribArray(4);

  uploadVec3(sm.vbo[5], 5, matTangent);
  uploadVec3(sm.vbo[6], 6, matBitangent);
  uploadVec3(sm.vbo[7], 7, decalTangent);
  uploadVec3(sm.vbo[8], 8, decalBitangent);

  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GltfModel::render(const Camera &camera, bool renderSelectedOnly, bool isWire,
                       bool isRenderTextureCoords, bool isRenderTexture, int currentTextureId,
                       const std::vector<std::unique_ptr<ImageTexture>> &textureList, float textureRadius,
                       const glm::vec2 &textureOffset, float textureTheta, PBRTexture *pbrTexture,
                       const std::vector<Light> &lights, bool flipNormals) {

  _program->use();

  _program->setMat4("projection_matrix", camera.projectionMatrixPointer());
  _program->setMat4("view_matrix", camera.viewMatrixPointer());
  auto modelMatrix = glm::identity<glm::mat4>();
  _program->setMat4("model_matrix", glm::value_ptr(modelMatrix));
  _program->setVec3("viewPos", glm::value_ptr(camera.eye()));

  // upload the enabled lights, packed contiguously into lights[0..numLights)
  int lightCount = 0;
  for (const Light &light : lights) {
    if (!light.enabled || lightCount >= MAX_LIGHTS)
      continue;
    glm::vec3 dir = light.direction();
    std::string base = "lights[" + std::to_string(lightCount) + "].";
    _program->setVec3((base + "direction").c_str(), glm::value_ptr(dir));
    _program->setVec3((base + "color").c_str(), glm::value_ptr(light.color));
    _program->setFloat((base + "intensity").c_str(), light.intensity);
    ++lightCount;
  }
  _program->setInt("numLights", lightCount);

  _program->setInt("isRenderTextureCoords", isRenderTextureCoords);
  _program->setInt("flipNormals", flipNormals);

  // decal placement transform (shared by every sub-mesh; only the active one samples it)
  _program->setFloat("textureRadius", textureRadius);
  _program->setVec2("textureOffset", glm::value_ptr(textureOffset));
  _program->setFloat("textureTheta", textureTheta);

  if (isWire)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  for (int i = 0; i < static_cast<int>(_subMeshes.size()); ++i)
    renderSubMesh(_subMeshes[i], i == _activeSubMesh, currentTextureId, pbrTexture, isRenderTextureCoords);

  if (isWire)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glBindVertexArray(0);
  Program::unUse();

  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR)
    ERROR("GltfModel Rendering Error: {}", err);
}

void GltfModel::renderSubMesh(SubMesh &sm, bool isActive, int currentTextureId, PBRTexture *pbrTexture,
                              bool isRenderTextureCoords) {

  const Material &mat = sm.material;

  auto bind = [&](unsigned int unit, const std::unique_ptr<ImageTexture> &tex, const char *name) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, tex ? tex->id() : _whiteTexture);
    _program->setInt(name, static_cast<int>(unit));
  };

  bind(0, mat.baseColor, "baseColorTex");
  bind(1, mat.metallicRoughness, "metallicRoughnessTex");
  bind(2, mat.normal, "normalTex");
  bind(3, mat.emissive, "emissiveTex");
  bind(4, mat.occlusion, "occlusionTex");

  _program->setVec3("baseColorFactor", glm::value_ptr(glm::vec3(mat.baseColorFactor)));
  _program->setFloat("metallicFactor", mat.metallicFactor);
  _program->setFloat("roughnessFactor", mat.roughnessFactor);
  _program->setVec3("emissiveFactor", glm::value_ptr(mat.emissiveFactor));
  _program->setInt("hasNormalTex", mat.normal != nullptr);

  const bool hasDecal = isActive && pbrTexture != nullptr;
  _program->setInt("isActiveSubMesh", isActive);
  _program->setInt("hasDecal", hasDecal);

  // defaults: no displacement, no subdivision. setupUniforms() overrides heightMode/tessLevel
  // for the active sub-mesh that carries the decal.
  _program->setInt("heightMode", 0);
  _program->setFloat("tessLevel", 1.0f);

  if (hasDecal) {
    // bind the editor's PBR decal on units 5..8 (basecolor/normal/height/mask) and
    // reuse the PBRTexture uniform plumbing (heightMode / tessLevel / decalHeightScale).
    pbrTexture->setupUniforms(*_program, 5,
                              PBRTexture::PBRTextureLocation{"decal.basecolor", "decal.normal",
                                                             "decalHeightMap", "decalRoughness", "decalMask",
                                                             "decalHeightScale"});
  }

  glBindVertexArray(sm.vao);
  glDrawArrays(GL_PATCHES, 0, sm.elementCount);
}

std::optional<glm::vec3> GltfModel::hit(const Camera &camera, const glm::vec2 &ndcPos) const {
  glm::vec4 rayClip(ndcPos, -1.0f, 1.0f);
  const glm::mat4 &proj = camera.projectionMatrix();
  const glm::mat4 &view = camera.viewMatrix();
  glm::vec4 rayEye = glm::inverse(proj) * rayClip;
  rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
  glm::vec3 rayDir = glm::normalize(glm::vec3(glm::inverse(view) * rayEye));
  const glm::vec3 &rayOrigin = camera.eye();

  std::optional<glm::vec3> best;
  float bestT = std::numeric_limits<float>::max();
  for (const SubMesh &sm : _subMeshes) {
    HitResult hit = sm.bvh.raycast(rayOrigin, rayDir);
    if (hit.faceIdx >= 0 && hit.t < bestT) {
      bestT = hit.t;
      best = hit.hitPoint;
    }
  }
  return best;
}

bool GltfModel::select(const glm::vec3 &hitPoint, int radius, bool isAdd) {
  if (_subMeshes.empty())
    return false;

  // pick the sub-mesh whose surface is closest to the hit point
  int target = -1;
  float bestDist2 = std::numeric_limits<float>::max();
  ClosestPointResult targetResult;
  for (int i = 0; i < static_cast<int>(_subMeshes.size()); ++i) {
    ClosestPointResult r = _subMeshes[i].bvh.closestPoint(hitPoint);
    if (r.faceIdx >= 0 && r.dist2 < bestDist2) {
      bestDist2 = r.dist2;
      target = i;
      targetResult = r;
    }
  }
  if (target < 0)
    return false;

  // switching active sub-mesh: only one mesh may carry a decal at a time
  if (target != _activeSubMesh) {
    if (_activeSubMesh >= 0) {
      _subMeshes[_activeSubMesh].selectedID.clear();
      updateSelectBuffer(_subMeshes[_activeSubMesh]);
    }
    _activeSubMesh = target;
  }

  SubMesh &sm = _subMeshes[_activeSubMesh];

  // BFS over face neighbours up to `radius`, mirroring Model::select
  std::unordered_set<int> visited;
  std::queue<std::pair<MyMesh::FaceHandle, int>> queue;
  queue.emplace(sm.mesh.face_handle(targetResult.faceIdx), 0);
  visited.insert(targetResult.faceIdx);

  bool dirty = false;
  while (!queue.empty()) {
    auto [fh, depth] = queue.front();
    queue.pop();

    auto flag = sm.selectedID.find(fh.idx());
    if (isAdd && flag == sm.selectedID.end()) {
      dirty = true;
      sm.selectedID.insert(fh.idx());
    } else if (!isAdd && flag != sm.selectedID.end()) {
      dirty = true;
      sm.selectedID.erase(flag);
    }

    if (depth >= radius)
      continue;

    for (const auto &neighbor : sm.mesh.ff_range(fh)) {
      if (!sm.mesh.is_valid_handle(neighbor) || visited.contains(neighbor.idx()))
        continue;
      visited.insert(neighbor.idx());
      queue.emplace(neighbor, depth + 1);
    }
  }

  return dirty;
}

void GltfModel::clearSelect() {
  if (_activeSubMesh < 0)
    return;
  _subMeshes[_activeSubMesh].selectedID.clear();
  updateSelectBuffer(_subMeshes[_activeSubMesh]);
}

void GltfModel::solve(SolveUV::SolvingMode solvingMode, std::optional<glm::vec3> hitPoint) {
  if (_activeSubMesh < 0)
    return;
  SubMesh &sm = _subMeshes[_activeSubMesh];
  if (sm.selectedID.empty())
    return;

  SolveUV::Solve(solvingMode, sm.selectedID, sm.mesh, hitPoint);
  SolveUV::calculateTB(sm.mesh);
  updateDecalBuffers(sm);
}

void GltfModel::updateTextureInfo(const TextureEditor &editor) {
  if (_activeSubMesh < 0)
    return;
  updateSelectBuffer(_subMeshes[_activeSubMesh]);
}

void GltfModel::updateSelectBuffer(SubMesh &sm) {
  // sl marker: 0 on selected faces (decal shows here), -1 elsewhere.
  std::vector<GLint> sl(sm.elementCount, -1);
  for (unsigned int f : sm.selectedID) {
    int base = static_cast<int>(f) * 3;
    if (base + 2 < sm.elementCount) {
      sl[base] = sl[base + 1] = sl[base + 2] = 0;
    }
  }
  glBindBuffer(GL_ARRAY_BUFFER, sm.vbo[4]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(sl.size() * sizeof(GLint)), sl.data(), GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GltfModel::updateDecalBuffers(SubMesh &sm) {
  std::vector<glm::vec2> uvDecal;
  std::vector<glm::vec3> decalTangent, decalBitangent;
  uvDecal.reserve(sm.elementCount);
  decalTangent.reserve(sm.elementCount);
  decalBitangent.reserve(sm.elementCount);

  for (const MyMesh::FaceHandle &fh : sm.mesh.faces()) {
    for (const MyMesh::VertexHandle &vh : sm.mesh.fv_range(fh)) {
      uvDecal.emplace_back(Utils::toGlm(sm.mesh.texcoord2D(vh)));
      const OpenMesh::Vec3f &t = sm.mesh.data(vh).tangent;
      const OpenMesh::Vec3f &b = sm.mesh.data(vh).bitangent;
      decalTangent.emplace_back(t[0], t[1], t[2]);
      decalBitangent.emplace_back(b[0], b[1], b[2]);
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, sm.vbo[3]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(uvDecal.size() * sizeof(glm::vec2)), uvDecal.data(),
               GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, sm.vbo[7]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(decalTangent.size() * sizeof(glm::vec3)),
               decalTangent.data(), GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, sm.vbo[8]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(decalBitangent.size() * sizeof(glm::vec3)),
               decalBitangent.data(), GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

std::vector<TextureLine> GltfModel::getSelectedTextureLines() {
  if (_activeSubMesh < 0)
    return {};
  SubMesh &sm = _subMeshes[_activeSubMesh];
  if (sm.selectedID.empty())
    return {};

  std::vector<TextureLine> result;
  std::unordered_set<int> selectedHF;

  for (const unsigned int &i : sm.selectedID) {
    for (const MyMesh::HalfedgeHandle &j : sm.mesh.fh_range(sm.mesh.face_handle(i))) {
      if (selectedHF.contains(j.idx()))
        continue;

      const MyMesh::VertexHandle &to_v = sm.mesh.to_vertex_handle(j);
      const MyMesh::VertexHandle &from_v = sm.mesh.from_vertex_handle(j);

      const MyMesh::TexCoord2D &to_tex = sm.mesh.texcoord2D(to_v);
      const MyMesh::TexCoord2D &from_tex = sm.mesh.texcoord2D(from_v);
      if (to_tex[0] < 0.0f || to_tex[0] > 1.0f || to_tex[1] < 0.0f || to_tex[1] > 1.0f || from_tex[0] < 0.0f ||
          from_tex[0] > 1.0f || from_tex[1] < 0.0f || from_tex[1] > 1.0f)
        break;

      result.push_back({{to_tex[0], to_tex[1]}, {from_tex[0], from_tex[1]}});
      selectedHF.insert(j.idx());
    }
  }

  return result;
}
