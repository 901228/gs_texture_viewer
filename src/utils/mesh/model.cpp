#include "model.hpp"

#include <filesystem>
#include <queue>
#include <unordered_set>

#include <glad/gl.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/trigonometric.hpp>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Tools/Utils/getopt.h>

#if not _MSC_VER
#include <OpenMesh/Core/IO/reader/OBJReader.hh>
#include <OpenMesh/Core/IO/writer/OBJWriter.hh>
#endif

#include "../camera/camera.hpp"
#include "../utils.hpp"

Model::Model()
    : _renderingProgram(std::make_unique<Program>(
          PROJECT_DIR "/src/shaders/shader.vert", PROJECT_DIR "/src/shaders/shader.frag",
          PROJECT_DIR "/src/shaders/shader.geom", PROJECT_DIR "/src/shaders/shader.tesc",
          PROJECT_DIR "/src/shaders/shader.tese")),
      _selectedID(std::make_unique<std::unordered_set<unsigned int>>()) {

  _mesh.request_vertex_status();
  _mesh.request_edge_status();
  _mesh.request_face_status();
}

Model::Model(char *path) : Model() { loadModel(path); }

Model::~Model() {

  if (_vertexBufferObject != nullptr)
    glDeleteBuffers(4, _vertexBufferObject);

  if (_vertexArrayObject != 0)
    glDeleteVertexArrays(1, &_vertexArrayObject);

  _mesh.release_vertex_status();
  _mesh.release_edge_status();
  _mesh.release_face_status();
}

size_t Model::n_faces() const { return _mesh.n_faces(); }
size_t Model::n_vertices() const { return _mesh.n_faces() * 3; }

glm::vec3 Model::center() const { return Utils::center(_boxmin, _boxmax); }

void Model::use() {
  _renderingProgram->use();
  glBindVertexArray(_vertexArrayObject);
}

void Model::unUse() {

  glBindVertexArray(0);
  Program::unUse();
}

bool Model::loadModel(const char *path) {

  this->use();

#if not _MSC_VER
  OpenMesh::IO::_OBJReader_(); // NOLINT(bugprone-unused-raii)
  OpenMesh::IO::_OBJWriter_(); // NOLINT(bugprone-unused-raii)
#endif

  std::string absPath = std::filesystem::absolute(path).string();

  OpenMesh::IO::Options opt;
  if (OpenMesh::IO::read_mesh(_mesh, absPath, opt)) {
    // If the file did not provide vertex normals and mesh has vertex normal
    // ,then calculate them
    if (!opt.check(OpenMesh::IO::Options::VertexNormal) && _mesh.has_vertex_normals()) {

      _mesh.request_face_normals();
      _mesh.update_normals();
      _mesh.release_face_normals();
    }

    initMesh();

    return true;
  } else
    ERROR("fail to load file from: {}", absPath);

  unUse();
  return false;
}

void Model::initMesh() {

  _bvh.build(_mesh);

  _vertices.clear();

  std::vector<glm::vec3> normals;
  for (const MyMesh::FaceHandle &i : _mesh.faces()) {
    for (const MyMesh::VertexHandle &j : _mesh.fv_range(i)) {
      glm::vec3 v = Utils::toGlm(_mesh.point(j));
      _vertices.emplace_back(v);
      normals.emplace_back(Utils::toGlm(_mesh.normal(j)));
      _mesh.set_texcoord2D(j, {0, 0});

      _boxmin = glm::min(_boxmin, v);
      _boxmax = glm::max(_boxmax, v);
    }
  }

  size_t vertexCount = n_faces() * 3;
  std::vector<GLint> selectIdx(vertexCount, -1);
  std::vector<glm::vec2> textureCoord = std::vector<glm::vec2>(vertexCount, {0, 0});
  std::vector<glm::vec3> tangent = std::vector<glm::vec3>(vertexCount, {0, 0, 0});
  std::vector<glm::vec3> bitangent = std::vector<glm::vec3>(vertexCount, {0, 0, 0});

  glGenVertexArrays(1, &_vertexArrayObject);
  glBindVertexArray(_vertexArrayObject);

  _vertexBufferObject = new GLuint[6];
  glGenBuffers(6, _vertexBufferObject);

  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[0]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(_vertices.size() * sizeof(glm::vec3)), _vertices.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[1]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(normals.size() * sizeof(glm::vec3)), normals.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[2]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(textureCoord.size() * sizeof(glm::vec2)),
               textureCoord.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(2);

  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[3]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(selectIdx.size() * sizeof(GLint)), selectIdx.data(),
               GL_STATIC_DRAW);
  // use "glVertexAttrib 'I' Pointer" for int
  glVertexAttribIPointer(3, 1, GL_INT, 0, nullptr);
  glEnableVertexAttribArray(3);

  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[4]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(tangent.size() * sizeof(glm::vec3)), tangent.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(4);

  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[5]);
  glBufferData(GL_ARRAY_BUFFER, static_cast<long>(bitangent.size() * sizeof(glm::vec3)), bitangent.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(5);

  _elementAmount = static_cast<GLsizei>(n_faces() * 3);

  INFO("faces count: {}", n_faces());
  INFO("elementAmount: {}", _elementAmount);

  unUse();
}

void Model::setupUniforms(const Camera &camera, bool isWire, bool isRenderTextureCoords, bool isRenderTexture,
                          int currentTextureId, const std::vector<std::unique_ptr<ImageTexture>> &textureList,
                          float textureRadius, const glm::vec2 &textureOffset, float textureTheta,
                          const glm::vec3 &lightDirection, float lightIntensity) {

  _renderingProgram->setMat4("projection_matrix", camera.projectionMatrixPointer());
  _renderingProgram->setMat4("view_matrix", camera.viewMatrixPointer());

  // model matrix
  auto model = glm::identity<glm::mat4>();
  _renderingProgram->setMat4("model_matrix", glm::value_ptr(model));

  _renderingProgram->setVec3("viewPos", glm::value_ptr(camera.eye()));

  _renderingProgram->setInt("isRenderWire", isWire);
  _renderingProgram->setInt("isRenderTextureCoords", isRenderTextureCoords);

  // light
  //   _renderingProgram->setVec3("dirLight[0].direction",
  //                    glm::value_ptr(glm::vec3(6.0f, 5.0f, 10.0f)));
  _renderingProgram->setVec3("dirLight[0].direction", glm::value_ptr(lightDirection));
  _renderingProgram->setVec3("dirLight[0].color", glm::value_ptr(glm::vec3(1.0f, 1.0f, 1.0f)));
  _renderingProgram->setFloat("dirLight[0].intensity", lightIntensity);
  // _renderingProgram->setVec3("dirLight[1].direction", glm::value_ptr(glm::vec3(-12.0f, -10.0f, -20.0f)));
  // _renderingProgram->setVec3("dirLight[1].color", glm::value_ptr(glm::vec3(1.0f, 1.0f, 1.0f)));

  _renderingProgram->setInt("isEditTexture", currentTextureId != -1);
  _renderingProgram->setInt("currentSL", currentTextureId);

  _renderingProgram->setInt("isRenderTexture", isRenderTexture);
  _renderingProgram->setFloat("textureRadius", textureRadius);
  _renderingProgram->setVec2("textureOffset", glm::value_ptr(textureOffset));
  _renderingProgram->setFloat("textureTheta", glm::radians(textureTheta));

  // for (int i = 0; i < textureList.size(); i++) {
  //   textureList[i]->setupUniforms(*_renderingProgram, i);
  // }

  _renderingProgram->setFloat("tessLevel", GL_MAX_TESS_GEN_LEVEL);
}

void Model::render(const Camera &camera, bool renderSelectedOnly, bool isWire, bool isRenderTextureCoords,
                   bool isRenderTexture, int currentTextureId,
                   const std::vector<std::unique_ptr<ImageTexture>> &textureList, float textureRadius,
                   const glm::vec2 &textureOffset, float textureTheta, PBRTexture *pbrTexture,
                   const glm::vec3 &lightDirection, float lightIntensity) {

  use();
  setupUniforms(camera, isWire, isRenderTextureCoords, isRenderTexture, currentTextureId, textureList,
                textureRadius, textureOffset, textureTheta, lightDirection, lightIntensity);

  if (pbrTexture != nullptr)
    pbrTexture->setupUniforms(*_renderingProgram);

  if (!_selectedID->empty()) {
    _renderingProgram->setInt("isRenderSelect", true);

    std::vector<GLint> first;
    for (const unsigned int i : *_selectedID)
      first.push_back(static_cast<int>(i * 3));
    std::vector<GLsizei> count(_selectedID->size(), 3);
    glMultiDrawArrays(GL_PATCHES, &first[0], &count[0], static_cast<GLsizei>(_selectedID->size()));

    _renderingProgram->setInt("isRenderSelect", false);
  }

  if (!renderSelectedOnly) {
    glDrawArrays(GL_PATCHES, 0, _elementAmount);
  }

  unUse();

  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) {
    ERROR("Model Rendering Error: {}", err);
  }
}

HitResult Model::select(const Camera &camera, float width, float height, const glm::vec2 &mousePos) const {

  float x = 2.0f * (mousePos.x / width) - 1.0f;
  float y = 1.0f - 2.0f * (mousePos.y / height);
  glm::vec4 rayClip(x, y, -1.0f, 1.0f);

  const glm::mat4 &proj = camera.projectionMatrix();
  const glm::mat4 &view = camera.viewMatrix();
  glm::vec4 rayEye = glm::inverse(proj) * rayClip;
  rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
  glm::vec3 rayDir = glm::normalize(glm::vec3(glm::inverse(view) * rayEye));
  const glm::vec3 &rayOrigin = camera.eye();

  HitResult hit = _bvh.raycast(rayOrigin, rayDir);
  return hit;
}

bool Model::selectRadius(int id, int radius, bool isAdd) {

  if (id < 0 || id >= n_faces())
    return false;

  std::unordered_set<int> visited;

  // {faceHandle, depth}
  std::queue<std::pair<MyMesh::FaceHandle, int>> queue;
  queue.emplace(_mesh.face_handle(id), 0);
  visited.insert(id);

  bool dirty = false;

  while (!queue.empty()) {
    auto [fh, depth] = queue.front();
    queue.pop();

    auto flag = _selectedID->find(fh.idx());
    if (isAdd && flag == _selectedID->end()) {
      dirty = true;
      _selectedID->insert(fh.idx());
    } else if (!isAdd && flag != _selectedID->end()) {
      dirty = true;
      _selectedID->erase(flag);
    }

    if (depth >= radius)
      continue;

    for (const auto &neighbor : _mesh.ff_range(fh)) {

      if (!_mesh.is_valid_handle(neighbor) || visited.contains(neighbor.idx()))
        continue;

      visited.insert(neighbor.idx());
      queue.emplace(neighbor, depth + 1);
    }
  }

  return dirty;
}

void Model::clearSelect() { _selectedID->clear(); }

void Model::calculateParameterization(SolveUV::SolvingMode solvingMode, float angle) {
  if (_selectedID->empty())
    return;

  SolveUV::Solve(solvingMode, *_selectedID, angle, _mesh);
  updateTexcoordVAO();
}

void Model::updateTexcoordVAO() {

  const size_t vertexCount = n_faces() * 3;

  // texcoord
  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[2]);
  glm::vec2 *texcoord = reinterpret_cast<glm::vec2 *>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));

  // tangent
  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[4]);
  glm::vec3 *tangent = reinterpret_cast<glm::vec3 *>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));

  // bitangent
  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[5]);
  glm::vec3 *bitangent = reinterpret_cast<glm::vec3 *>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));

  for (const unsigned int &faceId : *_selectedID) {
    MyMesh::FaceHandle fh = _mesh.face_handle(faceId);

    int index = faceId * 3;
    for (const MyMesh::VertexHandle vh : _mesh.fv_range(fh)) {

      const MyMesh::TexCoord2D _tc = _mesh.texcoord2D(vh);
      const OpenMesh::Vec3f _t = _mesh.data(vh).tangent;
      const OpenMesh::Vec3f _bt = _mesh.data(vh).bitangent;

      texcoord[index] = {_tc[0], _tc[1]};
      tangent[index] = {_t[0], _t[1], _t[2]};
      bitangent[index] = {_bt[0], _bt[1], _bt[2]};
      index++;
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[2]);
  glUnmapBuffer(GL_ARRAY_BUFFER);

  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[4]);
  glUnmapBuffer(GL_ARRAY_BUFFER);

  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[5]);
  glUnmapBuffer(GL_ARRAY_BUFFER);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Model::updateTexId(TextureEditor &textureEditor) {
  // TODO: update texture ID buffer
}

std::vector<TextureLine> Model::getSelectedTextureLines() {
  if (_selectedID->empty())
    return {};

  std::vector<TextureLine> result;
  std::unordered_set<int> selectedHF;

  for (const unsigned int &i : *_selectedID) {
    for (const MyMesh::HalfedgeHandle &j : _mesh.fh_range(_mesh.face_handle(i))) {

      if (selectedHF.contains(j.idx()))
        continue;

      const MyMesh::VertexHandle &to_v = _mesh.to_vertex_handle(j);
      const MyMesh::VertexHandle &from_v = _mesh.from_vertex_handle(j);

      const MyMesh::TexCoord2D &to_v_tex = _mesh.texcoord2D(to_v);
      const MyMesh::TexCoord2D &from_v_tex = _mesh.texcoord2D(from_v);
      if (to_v_tex[0] < 0.0f || to_v_tex[0] > 1.0f || to_v_tex[1] < 0.0f || to_v_tex[1] > 1.0f ||
          from_v_tex[0] < 0.0f || from_v_tex[0] > 1.0f || from_v_tex[1] < 0.0f || from_v_tex[1] > 1.0f)
        break;

      result.push_back({{to_v_tex[0], to_v_tex[1]}, {from_v_tex[0], from_v_tex[1]}});
      selectedHF.insert(j.idx());
    }
  }

  return result;
}

std::vector<std::pair<unsigned int, std::pair<float, float>>> Model::getSelectedTextureCoords() const {
  if (_selectedID->empty())
    return {};

  std::vector<std::pair<unsigned int, std::pair<float, float>>> result;
  std::unordered_set<int> selectedVertices;

  for (const unsigned int &i : *_selectedID) {
    for (const MyMesh::VertexHandle &j : _mesh.fv_range(_mesh.face_handle(i))) {

      if (selectedVertices.contains(j.idx()))
        continue;

      const MyMesh::TexCoord2D &v_tex = _mesh.texcoord2D(j);
      if (v_tex[0] < 0.0f || v_tex[0] > 1.0f || v_tex[1] < 0.0f || v_tex[1] > 1.0f)
        break;

      result.push_back({0, {v_tex[0], v_tex[1]}});
      selectedVertices.insert(j.idx());
    }
  }

  return result;
}
