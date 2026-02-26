#ifndef PLY_HPP
#define PLY_HPP
#pragma once

#include <cfloat>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#define _USE_MATH_DEFINES
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include "../utils/logger.hpp"
#include "../utils/mesh/mesh.hpp"

#define MAX_SH_DEGREE 3
#define MAX_SH_COEFF ((MAX_SH_DEGREE + 1) * (MAX_SH_DEGREE + 1))

struct Pos {
  float x, y, z;
};
template <int D> struct SHs {
  float shs[(D + 1) * (D + 1) * 3];
};
struct Scale {
  float scale[3];
};
struct Rot {
  float rot[4];
};
template <int D> struct RichPoint {
  Pos pos{};
  float n[3]{};
  SHs<D> shs{};
  float opacity{};
  Scale scale{};
  Rot rot{};
};

struct Face {
  int vertex_indices[3];
};

struct Coords {
  float c[3];
};
template <int D> struct RichAppearancePoint {
  Pos pos{};
  float n[3]{};
  Coords c{};
  float d{};
  float faceId{};
  SHs<D> shs{};
  float opacity{};
  Scale scale{};
  Rot rot{};
};

inline float sigmoid(const float m1) { return 1.0f / (1.0f + std::exp(-m1)); }
inline float inverse_sigmoid(const float m1) { return std::log(m1 / (1.0f - m1)); }
template <int N> inline std::array<float, N> softmax(std::array<float, N> x) {
  float max_val = *std::max_element(x.begin(), x.end());
  float sum = 0.0f;
  for (int i = 0; i < N; i++) {
    x[i] = std::exp(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < N; i++)
    x[i] /= sum;
  return x;
}

namespace {

template <typename PointType>
void findBox(const int count, const std::vector<PointType> &points, glm::vec3 &minn, glm::vec3 &maxx) {

  // Gaussians are done training, they won't move anymore. Arrange
  // them according to 3D Morton order. This means better cache
  // behavior for reading Gaussians that end up in the same tile
  // (close in 3D --> close in 2D).
  minn = {FLT_MAX, FLT_MAX, FLT_MAX};
  maxx = -minn;
  for (int i = 0; i < count; i++) {
    glm::vec3 curr = {points[i].pos.x, points[i].pos.y, points[i].pos.z};
    maxx = glm::max(maxx, curr);
    minn = glm::min(minn, curr);
  }
}

template <typename PointType>
std::vector<std::pair<glm::uint64_t, int>> sort(const int count, const std::vector<PointType> &points,
                                                const glm::vec3 &minn, const glm::vec3 &maxx) {

  std::vector<std::pair<uint64_t, int>> mapp(count);
  // NOLINTBEGIN(hicpp-signed-bitwise)
  for (int i = 0; i < count; i++) {
    glm::vec3 curr = {points[i].pos.x, points[i].pos.y, points[i].pos.z};

    // normalize to [0,1]
    glm::vec3 rel = (curr - minn) / (maxx - minn);

    // scale to 21-bit grid
    glm::vec3 scaled = float((1 << 21) - 1) * rel;

    glm::ivec3 xyz{scaled};

    uint64_t code = 0;
    for (int i = 0; i < 21; i++) {
      code |= ((uint64_t(xyz.x & (1 << i))) << (2 * i + 0));
      code |= ((uint64_t(xyz.y & (1 << i))) << (2 * i + 1));
      code |= ((uint64_t(xyz.z & (1 << i))) << (2 * i + 2));
    }

    mapp[i].first = code;
    mapp[i].second = i;
  }
  // NOLINTEND(hicpp-signed-bitwise)
  auto sorter = [](const std::pair<uint64_t, int> &a, const std::pair<uint64_t, int> &b) {
    return a.first < b.first;
  };
  std::sort(mapp.begin(), mapp.end(), sorter);

  return mapp;
}

template <int D>
void parseData(const int count, const std::vector<RichPoint<D>> &points,
               const std::vector<std::pair<glm::uint64_t, int>> &mapp, std::vector<Pos> &pos,
               std::vector<SHs<3>> &shs, std::vector<float> &opacities, std::vector<Scale> &scales,
               std::vector<Rot> &rot) {

  // Resize our SoA data
  pos.resize(count);
  shs.resize(count);
  scales.resize(count);
  rot.resize(count);
  opacities.resize(count);

  // Move data from AoS to SoA
  int SH_N = (D + 1) * (D + 1);
  for (int k = 0; k < count; k++) {
    int i = mapp[k].second;
    pos[k] = points[i].pos;

    // Normalize quaternion
    float length2 = 0;
    for (int j = 0; j < 4; j++)
      length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
    float length = std::sqrt(length2);
    for (int j = 0; j < 4; j++)
      rot[k].rot[j] = points[i].rot.rot[j] / length;

    // Exponentiate scale
    for (int j = 0; j < 3; j++)
      scales[k].scale[j] = std::exp(points[i].scale.scale[j]);

    // Activate alpha
    opacities[k] = sigmoid(points[i].opacity);

    shs[k].shs[0] = points[i].shs.shs[0];
    shs[k].shs[1] = points[i].shs.shs[1];
    shs[k].shs[2] = points[i].shs.shs[2];
    for (int j = 1; j < SH_N; j++) {
      shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
      shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
      shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
    }
  }
}

template <int D>
void parseAppearanceData(const int count, const std::vector<RichAppearancePoint<D>> &points,
                         const std::vector<std::pair<glm::uint64_t, int>> &mapp, std::vector<Pos> &pos,
                         std::vector<SHs<3>> &shs, std::vector<float> &opacities, std::vector<Scale> &scales,
                         std::vector<Rot> &rot, std::vector<Coords> &coords, std::vector<float> &faceIds) {

  // Resize our SoA data
  pos.resize(count);
  shs.resize(count);
  scales.resize(count);
  rot.resize(count);
  opacities.resize(count);
  coords.resize(count);
  faceIds.resize(count);

  // Move data from AoS to SoA
  int SH_N = (D + 1) * (D + 1);
  for (int k = 0; k < count; k++) {
    int i = mapp[k].second;
    pos[k] = points[i].pos;

    // Normalize quaternion
    float length2 = 0;
    for (int j = 0; j < 4; j++)
      length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
    float length = std::sqrt(length2);
    for (int j = 0; j < 4; j++)
      rot[k].rot[j] = points[i].rot.rot[j] / length;

    // Exponentiate scale
    for (int j = 0; j < 3; j++)
      scales[k].scale[j] = std::exp(points[i].scale.scale[j]);

    // Activate alpha
    opacities[k] = sigmoid(points[i].opacity);

    shs[k].shs[0] = points[i].shs.shs[0];
    shs[k].shs[1] = points[i].shs.shs[1];
    shs[k].shs[2] = points[i].shs.shs[2];
    for (int j = 1; j < SH_N; j++) {
      shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
      shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
      shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
    }

    const auto c = softmax<3>({points[i].c.c[0], points[i].c.c[1], points[i].c.c[2]});
    coords[k] = {c[0], c[1], c[2]};
    faceIds[k] = points[i].faceId;
  }
}
} // namespace

// Load the Gaussians from the given file.
template <int D>
inline int loadPly(const char *filename, std::vector<Pos> &pos, std::vector<SHs<3>> &shs,
                   std::vector<float> &opacities, std::vector<Scale> &scales, std::vector<Rot> &rot,
                   glm::vec3 &minn, glm::vec3 &maxx) {
  std::ifstream infile(filename, std::ios_base::binary);

  if (!infile.good())
    throw std::runtime_error(std::format("Unable to find model's PLY file, attempted:\n{}\n", filename));

  // "Parse" header (it has to be a specific format anyway)
  std::string buff;
  std::getline(infile, buff); // ply
  std::getline(infile, buff); // format binary_little_endian 1.0

  std::string dummy;
  std::getline(infile, buff); // element vertex 49990
  std::stringstream ss(buff);
  int count;
  ss >> dummy >> dummy >> count;

  // Output number of Gaussians contained
  INFO("Loading {} Gaussian splats", count);

  while (std::getline(infile, buff))
    if (buff == "end_header")
      break;

  // Read all Gaussians at once (AoS)
  std::vector<RichPoint<D>> points(count);
  infile.read((char *)points.data(), count * sizeof(RichPoint<D>));

  findBox(count, points, minn, maxx);

  auto mapp = sort(count, points, minn, maxx);

  parseData(count, points, mapp, pos, shs, opacities, scales, rot);

  return count;
}

// load geometry gaussian
template <int D>
inline int loadGeometryPly(const char *filename, std::vector<Pos> &pos, std::vector<SHs<3>> &shs,
                           std::vector<float> &opacities, std::vector<Scale> &scales, std::vector<Rot> &rot,
                           std::vector<Face> &faces, glm::vec3 &minn, glm::vec3 &maxx) {
  std::ifstream infile(filename, std::ios_base::binary);

  if (!infile.good())
    throw std::runtime_error(std::format("Unable to find model's PLY file, attempted:\n{}\n", filename));

  // "Parse" header (it has to be a specific format anyway)
  std::string buff;
  std::getline(infile, buff); // ply
  std::getline(infile, buff); // format binary_little_endian 1.0

  std::string dummy;
  std::getline(infile, buff); // element vertex 49990
  std::stringstream ss(buff);
  int count;
  ss >> dummy >> dummy >> count;

  // Output number of Gaussians contained
  INFO("Loading {} Gaussian splats", count);

  // read until `end_header`, and record whether there is a `face` element
  int face_count = 0;
  bool has_face = false;
  while (std::getline(infile, buff)) {
    if (buff.find("element face") != std::string::npos) {
      // parse face count
      std::stringstream ss_face(buff);
      ss_face >> dummy >> dummy >> face_count;
      has_face = true;
    }
    if (buff == "end_header")
      break;
  }

  // read vertex data
  std::vector<RichPoint<D>> points(count);
  infile.read((char *)points.data(), count * sizeof(RichPoint<D>));

  // read face data if exists
  if (has_face && face_count > 0) {
    faces.resize(face_count);

    for (int i = 0; i < face_count; i++) {
      // read vertex count (1 byte)
      unsigned char vertex_count;
      infile.read((char *)&vertex_count, sizeof(unsigned char));

      if (vertex_count != 3) {
        WARN("Face {} has {} vertices (expected 3), skipping "
             "non-triangular face",
             i, (int)vertex_count);
      }

      // read vertex indices
      int indices[3];
      infile.read((char *)indices, static_cast<long>(vertex_count * sizeof(int)));

      // only save triangles
      if (vertex_count == 3) {
        faces[i].vertex_indices[0] = indices[0];
        faces[i].vertex_indices[1] = indices[1];
        faces[i].vertex_indices[2] = indices[2];
      } else {
        // if not triangle, fill invalid index
        faces[i].vertex_indices[0] = -1;
        faces[i].vertex_indices[1] = -1;
        faces[i].vertex_indices[2] = -1;
      }
    }

    INFO("Successfully loaded {} faces", face_count);
  }

  // find bounding box
  findBox(count, points, minn, maxx);

  // morton code sort
  auto mapp = sort(count, points, minn, maxx);

  // create mapping (old index -> new index)
  std::vector<int> old_to_new(count);
  for (int k = 0; k < count; k++) {
    int original_idx = mapp[k].second;
    old_to_new[original_idx] = k;
  }

  // update face vertex indices (from original index to new index)
  if (has_face && face_count > 0) {
    for (int i = 0; i < face_count; i++) {
      if (faces[i].vertex_indices[0] != -1) {
        // convert original index to new index
        faces[i].vertex_indices[0] = old_to_new[faces[i].vertex_indices[0]];
        faces[i].vertex_indices[1] = old_to_new[faces[i].vertex_indices[1]];
        faces[i].vertex_indices[2] = old_to_new[faces[i].vertex_indices[2]];
      }
    }
    INFO("Updated {} face indices to match sorted vertices", face_count);
  }

  // AoS to SoA
  parseData(count, points, mapp, pos, shs, opacities, scales, rot);

  return count;
}

template <int D>
inline int loadAppearancePly(const char *filename, std::vector<Pos> &pos, std::vector<SHs<3>> &shs,
                             std::vector<float> &opacities, std::vector<Scale> &scales, std::vector<Rot> &rot,
                             std::vector<Coords> &coords, std::vector<float> &faceIds, glm::vec3 &minn,
                             glm::vec3 &maxx) {
  std::ifstream infile(filename, std::ios_base::binary);

  if (!infile.good())
    throw std::runtime_error(std::format("Unable to find model's PLY file, attempted:\n{}\n", filename));

  // "Parse" header (it has to be a specific format anyway)
  std::string buff;
  std::getline(infile, buff); // ply
  std::getline(infile, buff); // format binary_little_endian 1.0

  std::string dummy;
  std::getline(infile, buff); // element vertex 49990
  std::stringstream ss(buff);
  int count;
  ss >> dummy >> dummy >> count;

  // Output number of Gaussians contained
  INFO("Loading {} Gaussian splats", count);

  while (std::getline(infile, buff))
    if (buff == "end_header")
      break;

  // Read all Gaussians at once (AoS)
  std::vector<RichAppearancePoint<D>> points(count);
  infile.read((char *)points.data(), count * sizeof(RichAppearancePoint<D>));

  findBox(count, points, minn, maxx);

  auto mapp = sort(count, points, minn, maxx);

  parseAppearanceData(count, points, mapp, pos, shs, opacities, scales, rot, coords, faceIds);

  return count;
}

inline bool createMeshFromGaussians(const std::vector<Pos> &pos, const std::vector<Face> &faces,
                                    MyMesh &mesh) {

  mesh.clear();
  mesh.request_vertex_normals();
  mesh.request_face_normals();
  mesh.request_vertex_texcoords2D();

  mesh.request_halfedge_normals();
  mesh.request_vertex_status();
  mesh.request_edge_status();
  mesh.request_face_status();
  mesh.request_halfedge_status();

  // add all vertices
  std::vector<MyMesh::VertexHandle> vertex_handles;
  vertex_handles.reserve(pos.size());

  for (auto p : pos) {
    MyMesh::Point point(p.x, p.y, p.z);
    OpenMesh::SmartVertexHandle vh = mesh.add_vertex(point);
    vertex_handles.push_back(vh);

    mesh.set_texcoord2D(vh, {0, 0}); // NOLINT(cppcoreguidelines-slicing)
  }

  INFO("Added {} vertices to mesh", pos.size());

  // add all faces
  int valid_faces = 0;
  int invalid_faces = 0;

  for (auto face : faces) {
    if (face.vertex_indices[0] == -1 || face.vertex_indices[1] == -1 || face.vertex_indices[2] == -1) {
      invalid_faces++;
      continue;
    }

    int idx0 = face.vertex_indices[0];
    int idx1 = face.vertex_indices[1];
    int idx2 = face.vertex_indices[2];

    if (idx0 < 0 || idx0 >= static_cast<int>(pos.size()) || idx1 < 0 ||
        idx1 >= static_cast<int>(pos.size()) || idx2 < 0 || idx2 >= static_cast<int>(pos.size())) {
      invalid_faces++;
      continue;
    }

    std::vector<MyMesh::VertexHandle> face_vhandles;
    face_vhandles.push_back(vertex_handles[idx0]);
    face_vhandles.push_back(vertex_handles[idx1]);
    face_vhandles.push_back(vertex_handles[idx2]);

    OpenMesh::SmartFaceHandle fh = mesh.add_face(face_vhandles);

    if (fh.is_valid()) {
      valid_faces++;
    } else {
      invalid_faces++;
    }
  }

  INFO("Added {} valid faces, skipped {} invalid faces", valid_faces, invalid_faces);

  mesh.update_normals();

  return valid_faces > 0;
}

// TODO: save ply

#endif // !PLY_HPP
