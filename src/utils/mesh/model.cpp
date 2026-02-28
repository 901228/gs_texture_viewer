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
    : _renderingProgram(std::make_unique<Program>(PROJECT_DIR "/src/shaders/shader.vert",
                                                  PROJECT_DIR "/src/shaders/shader.frag",
                                                  PROJECT_DIR "/src/shaders/shader.geom")),
      _selectingProgram(std::make_unique<Program>(PROJECT_DIR "/src/shaders/shader.vert",
                                                  PROJECT_DIR "/src/shaders/selecting.frag", "")),
      _selectedID(std::make_unique<std::set<unsigned int>>()) {

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

const Program &Model::use(bool isSelect) {

  if (isSelect) {
    _selectingProgram->use();
    glBindVertexArray(_vertexArrayObject);
    return *_selectingProgram;
  } else {
    _renderingProgram->use();
    glBindVertexArray(_vertexArrayObject);
    return *_renderingProgram;
  }
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

void Model::initMesh(bool toGL) {

  _vertices.clear();
  std::vector<GLfloat> normals;

  MyMesh::Normal n;
  MyMesh::Point v;
  for (const MyMesh::FaceHandle &i : _mesh.faces()) {
    for (const MyMesh::VertexHandle &j : _mesh.fv_range(i)) {
      n = _mesh.normal(j);
      v = _mesh.point(j);

      _vertices.push_back(v[0]);
      _vertices.push_back(v[1]);
      _vertices.push_back(v[2]);

      normals.push_back(n[0]);
      normals.push_back(n[1]);
      normals.push_back(n[2]);

      _mesh.set_texcoord2D(j, {0, 0});
    }
  }

  std::vector<GLint> selectIdx(n_faces() * 3, -1);
  std::vector<glm::vec2> textureCoord = std::vector<glm::vec2>(n_faces() * 3, {0, 0});

  // ===========================================================================================

  if (toGL) {

    glGenVertexArrays(1, &_vertexArrayObject);
    glBindVertexArray(_vertexArrayObject);

    _vertexBufferObject = new GLuint[4];
    glGenBuffers(4, _vertexBufferObject);

    glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[0]);
    glBufferData(GL_ARRAY_BUFFER, static_cast<long>(_vertices.size() * sizeof(GLfloat)), &_vertices[0],
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[1]);
    glBufferData(GL_ARRAY_BUFFER, static_cast<long>(normals.size() * sizeof(GLfloat)), &normals[0],
                 GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[2]);
    glBufferData(GL_ARRAY_BUFFER, static_cast<long>(textureCoord.size() * sizeof(glm::vec2)),
                 &textureCoord[0], GL_STATIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[3]);
    glBufferData(GL_ARRAY_BUFFER, static_cast<long>(selectIdx.size() * sizeof(GLint)), &selectIdx[0],
                 GL_STATIC_DRAW);
    // use "glVertexAttrib 'I' Pointer" for int
    glVertexAttribIPointer(3, 1, GL_INT, 0, nullptr);
    glEnableVertexAttribArray(3);

    _elementAmount = static_cast<GLsizei>(n_faces() * 3);

    INFO("faces count: {}", n_faces());
    INFO("elementAmount: {}", _elementAmount);

    unUse();
  }
}

void Model::setupUniformsCommon(const Program &program, const Camera &camera) {

  program.setMat4("projection_matrix", camera.projectionMatrixPointer());
  program.setMat4("view_matrix", camera.viewMatrixPointer());

  // model matrix
  auto model = glm::identity<glm::mat4>();
  program.setMat4("model_matrix", glm::value_ptr(model));
}

void Model::setupUniforms(const Camera &camera, bool isWire, bool isRenderTextureCoords, bool isRenderTexture,
                          int currentTextureId, const std::vector<std::unique_ptr<ImageTexture>> &textureList,
                          float textureRadius, const glm::vec2 &textureOffset, float textureTheta) {

  _renderingProgram->setVec3("viewPos", glm::value_ptr(camera.eye()));

  _renderingProgram->setInt("isRenderWire", isWire);
  _renderingProgram->setInt("isRenderTextureCoords", isRenderTextureCoords);

  // light
  //   _renderingProgram->setVec3("dirLight[0].direction",
  //                    glm::value_ptr(glm::vec3(6.0f, 5.0f, 10.0f)));
  _renderingProgram->setVec3("dirLight[0].direction", glm::value_ptr(glm::vec3(0.0f, -1.0f, 0.0f)));
  _renderingProgram->setVec3("dirLight[0].color", glm::value_ptr(glm::vec3(1.0f, 1.0f, 1.0f)));
  _renderingProgram->setVec3("dirLight[1].direction", glm::value_ptr(glm::vec3(-12.0f, -10.0f, -20.0f)));
  _renderingProgram->setVec3("dirLight[1].color", glm::value_ptr(glm::vec3(1.0f, 1.0f, 1.0f)));

  _renderingProgram->setInt("isEditTexture", currentTextureId != -1);
  _renderingProgram->setInt("currentSL", currentTextureId);

  _renderingProgram->setInt("isRenderTexture", isRenderTexture);
  _renderingProgram->setFloat("textureRadius", textureRadius);
  _renderingProgram->setVec2("textureOffset", glm::value_ptr(textureOffset));
  _renderingProgram->setFloat("textureTheta", glm::radians(textureTheta));

  for (int i = 0; i < textureList.size(); i++) {
    textureList[i]->setupUniforms(*_renderingProgram, i);
  }
}

void Model::render(const Camera &camera, bool isSelect, bool renderSelectedOnly, bool isWire,
                   bool isRenderTextureCoords, bool isRenderTexture, int currentTextureId,
                   const std::vector<std::unique_ptr<ImageTexture>> &textureList, float textureRadius,
                   const glm::vec2 &textureOffset, float textureTheta) {

  const Program &program = use(isSelect);
  setupUniformsCommon(program, camera);
  if (!isSelect) {
    setupUniforms(camera, isWire, isRenderTextureCoords, isRenderTexture, currentTextureId, textureList,
                  textureRadius, textureOffset, textureTheta);
  }

  if (!isSelect && !_selectedID->empty()) {
    _renderingProgram->setInt("isRenderSelect", true);

    std::vector<GLint> first;
    for (const unsigned int i : *_selectedID)
      first.push_back(static_cast<int>(i * 3));
    std::vector<GLsizei> count(_selectedID->size(), 3);
    glMultiDrawArrays(GL_TRIANGLES, &first[0], &count[0], static_cast<GLsizei>(_selectedID->size()));

    _renderingProgram->setInt("isRenderSelect", false);
  }

  if (!renderSelectedOnly) {
    glDrawArrays(GL_TRIANGLES, 0, _elementAmount);
  }

  unUse();

  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) {
    ERROR("Model Rendering Error: {}", err);
  }
}

std::tuple<int, glm::vec3> Model::select(const Camera &camera, float width, float height,
                                         const glm::vec2 &mousePos) {

  float x = 2.0f * (mousePos.x / width) - 1.0f;
  float y = 1.0f - 2.0f * (mousePos.y / height);
  glm::vec4 rayClip(x, y, -1.0f, 1.0f);

  const glm::mat4 &proj = camera.projectionMatrix();
  const glm::mat4 &view = camera.viewMatrix();
  glm::vec4 rayEye = glm::inverse(proj) * rayClip;
  rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
  glm::vec3 rayDir = glm::normalize(glm::vec3(glm::inverse(view) * rayEye));
  const glm::vec3 &rayOrigin = camera.eye();

  float minT = 1e9f;
  int faceIdx = -1;
  glm::vec3 hitPoint{};

  for (int i = 0; i < _vertices.size(); i += 9) {

    // get point of vertices
    glm::vec3 v0 = Utils::toGlm({_vertices[i], _vertices[i + 1], _vertices[i + 2]});
    glm::vec3 v1 = Utils::toGlm({_vertices[i + 3], _vertices[i + 4], _vertices[i + 5]});
    glm::vec3 v2 = Utils::toGlm({_vertices[i + 6], _vertices[i + 7], _vertices[i + 8]});

    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;
    glm::vec3 h = glm::cross(rayDir, e2);
    float a = glm::dot(e1, h);
    if (std::abs(a) < 1e-6)
      continue;
    float f = 1.0f / a;
    glm::vec3 s = rayOrigin - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f)
      continue;
    glm::vec3 q = glm::cross(s, e1);
    float v = f * glm::dot(rayDir, q);
    if (v < 0.0f || u + v > 1.0f)
      continue;
    float t = f * glm::dot(e2, q);

    if (t > 1e-4 && t < minT) {
      minT = t;
      faceIdx = i / 9;
      hitPoint = rayOrigin + rayDir * minT;
    }
  }

  if (faceIdx != -1) {
    hitPoint = rayOrigin + rayDir * minT;
  }

  return std::make_tuple(faceIdx, hitPoint);
}

void Model::selectRadius(int id, int radius, bool isAdd) {

  if (id < 0 || id >= n_faces())
    return;

  std::unordered_set<int> visited;

  // {faceHandle, depth}
  std::queue<std::pair<MyMesh::FaceHandle, int>> queue;
  queue.emplace(_mesh.face_handle(id), 0);
  visited.insert(id);

  while (!queue.empty()) {
    auto [fh, depth] = queue.front();
    queue.pop();

    if (isAdd) {
      _selectedID->insert(fh.idx());
    } else {
      auto flag = _selectedID->find(fh.idx());
      if (flag != _selectedID->end())
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
}

void Model::clearSelect() { _selectedID->clear(); }

void Model::calculateParameterization(SolveUV::SolvingMode solvingMode, float angle) {
  if (_selectedID->empty())
    return;

  SolveUV::Solve(solvingMode, *_selectedID, angle, _mesh);
  updateTexcoordVAO(false);
}

glm::vec2 *Model::updateTexcoordVAO(bool returnData) {

  // update texcoord VAO buffer
  glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject[2]);
  void *ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
  const size_t size = n_vertices();
  auto *data = new glm::vec2[size];
  memcpy(data, ptr, sizeof(glm::vec2) * size);

  int index = -1;
  for (const MyMesh::FaceHandle fh : _mesh.faces()) {

    for (const MyMesh::VertexHandle vh : _mesh.fv_range(fh)) {

      const MyMesh::TexCoord2D texCoord = _mesh.texcoord2D(vh);
      index++;

      if (data[index].x != -1 || data[index].y != -1) {

        if (_selectedID->find(fh.idx()) == _selectedID->end())
          continue;
      }
      data[index] = {texCoord[0], texCoord[1]};
    }
  }

  memcpy(ptr, data, sizeof(glm::vec2) * size);
  glUnmapBuffer(GL_ARRAY_BUFFER);

  if (returnData) {
    return data;
  } else {
    delete[] data;
    return nullptr;
  };
}

std::vector<TextureLine> Model::getSelectedTextureLines() {
  if (_selectedID->empty())
    return {};

  std::vector<TextureLine> result;
  std::set<int> selectedHF;

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
  std::set<int> selectedVertices;

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
