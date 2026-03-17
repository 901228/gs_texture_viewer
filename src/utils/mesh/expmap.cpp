#include "expmap.hpp"

#include <queue>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "../utils.hpp"

namespace {

inline int computeCenterFromFaces(const std::unordered_set<unsigned int> &selectedID, const MyMesh &mesh) {
  if (selectedID.empty())
    return -1;

  // Compute geometric center of all selected faces
  glm::vec3 center(0);
  int count = 0;
  for (unsigned int f : selectedID) {
    MyMesh::FaceHandle fh = mesh.face_handle(f);
    for (const MyMesh::VertexHandle &fv : mesh.fv_range(fh)) {
      center += Utils::toGlm(mesh.point(fv));
      count++;
    }
  }
  center /= float(count);

  // Find closest vertex to center (search within selected faces for efficiency)
  float minDist = std::numeric_limits<float>::max();
  int closestIdx = -1;
  std::unordered_set<int> checkedVertices;
  for (unsigned int f : selectedID) {
    MyMesh::FaceHandle fh = mesh.face_handle(f);
    for (const MyMesh::VertexHandle &fv : mesh.fv_range(fh)) {
      if (checkedVertices.contains(fv.idx()))
        continue;

      checkedVertices.insert(fv.idx());
      float dist = glm::length(Utils::toGlm(mesh.point(fv)) - center);
      if (dist < minDist) {
        minDist = dist;
        closestIdx = fv.idx();
      }
    }
  }
  return closestIdx;
}

// Tangent frame: origin + orthonormal basis (X, Y, Z where Z = normal)
struct TangentFrame {
  glm::vec3 origin = {0, 0, 0};
  glm::mat3 axes = glm::mat3(1); // columns: X, Y, Z

  TangentFrame() = default;

  inline TangentFrame(const glm::vec3 &pos, const glm::vec3 &normal) {

    origin = pos;
    glm::vec3 n = glm::normalize(normal);

    // Compute perpendicular vectors based on largest component
    glm::vec3 x;
    if (std::abs(n.x) >= std::abs(n.y) && std::abs(n.x) >= std::abs(n.z)) {
      x = glm::normalize(glm::vec3(-n.y, n.x, 0.0f));
    } else {
      x = glm::normalize(glm::vec3(0.0f, n.z, -n.y));
    }
    glm::vec3 y = glm::cross(n, x);

    // Store as columns: X, Y, Z
    axes = glm::mat3(x, y, n);
  }

  [[nodiscard]] inline glm::vec3 toLocal(const glm::vec3 &worldVec) const {
    return glm::transpose(axes) * worldVec;
  }

  inline void alignZAxis(const TangentFrame &target) {

    glm::vec3 fromZ = axes[2];
    glm::vec3 toZ = target.axes[2];

    glm::vec3 axis = glm::cross(fromZ, toZ);
    float sinAngle = glm::length(axis);
    float cosAngle = glm::dot(fromZ, toZ);

    // Already aligned or opposite
    if (sinAngle < 1e-6f) {
      if (cosAngle < 0) {
        // Opposite direction: rotate 180 around X axis
        axes = glm::mat3(axes[0], -axes[1], -axes[2]);
      }
      return;
    }

    axis = glm::normalize(axis);
    float angle = std::acos(glm::clamp(cosAngle, -1.0f, 1.0f));
    glm::mat3 rot = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, axis));
    axes = rot * axes;
  }
};

// Per-vertex data for ExpMap propagation
struct ExpMapVertex {
  enum class State { INACTIVE, ACTIVE, FROZEN };

  OpenMesh::VertexHandle vh;
  State state = State::INACTIVE;
  float distance = std::numeric_limits<float>::max();
  glm::vec2 surfaceVector = {0, 0}; // UV coordinates (geodesic)
  TangentFrame frame;
  OpenMesh::VertexHandle nearest; // parent in propagation tree
};

inline void propagateFrame(std::vector<ExpMapVertex> &vertices, ExpMapVertex &current, int seedIdx,
                           const MyMesh &mesh) {
  auto &nearest = vertices[current.nearest.idx()];

  // Initialize current frame from position and normal
  glm::vec3 curPos = Utils::toGlm(mesh.point(current.vh));
  glm::vec3 curNormal = Utils::toGlm(mesh.normal(current.vh));
  current.frame = TangentFrame(curPos, curNormal);

  // Get seed frame for rotation computation
  TangentFrame &seedFrame = vertices[seedIdx].frame;
  TangentFrame alignedSeedFrame = seedFrame;
  alignedSeedFrame.alignZAxis(nearest.frame);

  // Compute angle between X-axes after alignment
  glm::vec3 nearestX = nearest.frame.axes[0];
  glm::vec3 seedX = alignedSeedFrame.axes[0];

  float cosTheta = glm::clamp(glm::dot(nearestX, seedX), -1.0f, 1.0f);
  float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));

  // Sign correction using cross product
  glm::vec3 cross = glm::cross(nearestX, seedX);
  if (glm::dot(cross, nearest.frame.axes[2]) < 0) {
    sinTheta = -sinTheta;
  }

  // 2D rotation matrix (column-major for GLM, matching Wml row-major behavior)
  // Wml: mat * vec = (cos*x + sin*y, -sin*x + cos*y) - clockwise rotation
  glm::mat2 rot(cosTheta, -sinTheta, sinTheta, cosTheta);

  // Project current position into nearest's tangent plane (distance preserving)
  glm::vec3 nearestPos = Utils::toGlm(mesh.point(nearest.vh));
  glm::vec3 nearestNormal = nearest.frame.axes[2];

  // Orthogonal projection
  glm::vec3 projected = curPos - glm::dot(curPos - nearestPos, nearestNormal) * nearestNormal;

  // Scale to preserve distance from origin
  glm::vec3 origVec = curPos - nearestPos;
  glm::vec3 projVec = projected - nearestPos;
  float projLen = glm::length(projVec);
  float scale = (projLen > 1e-8f) ? glm::length(origVec) / projLen : 1.0f;
  glm::vec3 planePoint = nearestPos + scale * projVec;

  // Transform to local frame coordinates
  glm::vec3 localVec = nearest.frame.toLocal(planePoint - nearestPos);

  // Extract 2D surface vector (negate to point from nearest to current)
  glm::vec2 surfaceVec(-localVec.x, -localVec.y);

  // Accumulate with rotation
  current.surfaceVector = nearest.surfaceVector + rot * surfaceVec;

  // Error correction: if |surfaceVector| deviates >50% from geodesic distance,
  // reset to radial
  float vecLen = glm::length(current.surfaceVector);
  float distSq = current.distance * current.distance;
  if (distSq > 1e-12f) {
    float distError = std::abs(vecLen * vecLen / distSq - 1.0f);
    if (distError > 0.5f && vecLen > 1e-8f) {
      // Reset to radial direction, preserve geodesic distance magnitude
      current.surfaceVector = glm::normalize(current.surfaceVector) * current.distance;
    }
  }
}

} // namespace

namespace ExpMap {

void Solve(const std::unordered_set<unsigned int> &selectedID, MyMesh &originMesh) {

  if (selectedID.empty())
    return;

  // Auto-compute center from selection
  int centerVertex = computeCenterFromFaces(selectedID, originMesh);
  if (centerVertex < 0)
    return;

  auto seedVh = originMesh.vertex_handle(centerVertex);
  if (!seedVh.is_valid()) {
    return;
  }

  // Build set of vertices from selected faces
  std::unordered_set<int> selectedVertices;
  for (unsigned int f : selectedID) {
    MyMesh::FaceHandle fh = originMesh.face_handle(f);
    for (const MyMesh::VertexHandle &fv : originMesh.fv_range(fh)) {
      selectedVertices.insert(fv.idx());
    }
  }

  // Radius will be auto-computed from selection extent after Dijkstra
  float radius = 0.0f;

  // Initialize per-vertex data
  size_t numVerts = originMesh.n_vertices();
  std::vector<ExpMapVertex> vertices(numVerts);

  for (MyMesh::VertexHandle vh : originMesh.vertices()) {
    vertices[vh.idx()].vh = vh;
    vertices[vh.idx()].state = ExpMapVertex::State::INACTIVE;
    vertices[vh.idx()].distance = std::numeric_limits<float>::max();
  }

  // Initialize seed vertex
  int seedIdx = seedVh.idx();
  ExpMapVertex &seed = vertices[seedIdx];
  seed.state = ExpMapVertex::State::FROZEN;
  seed.distance = 0.0f;
  seed.surfaceVector = glm::vec2(0, 0);

  glm::vec3 seedPos = Utils::toGlm(originMesh.point(seedVh));
  glm::vec3 seedNormal = Utils::toGlm(originMesh.normal(seedVh));
  seed.frame = TangentFrame(seedPos, seedNormal);

  // Priority queue (min-heap by distance)
  auto cmp = [&vertices](int a, int b) { return vertices[a].distance > vertices[b].distance; };
  std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

  // Add seed neighbors to queue (only if in selected vertices)
  for (const MyMesh::HalfedgeHandle &voh : originMesh.voh_range(seedVh)) {
    auto nvh = originMesh.to_vertex_handle(voh);
    if (!selectedVertices.count(nvh.idx()))
      continue; // Skip non-selected
    auto &nv = vertices[nvh.idx()];
    nv.distance = glm::length(Utils::toGlm(originMesh.point(nvh)) - seedPos);
    nv.nearest = seedVh;
    nv.state = ExpMapVertex::State::ACTIVE;
    pq.push(nvh.idx());
  }

  // Dijkstra propagation
  while (!pq.empty()) {
    int curIdx = pq.top();
    pq.pop();

    auto &cur = vertices[curIdx];
    if (cur.state == ExpMapVertex::State::FROZEN) {
      continue; // Already processed
    }
    cur.state = ExpMapVertex::State::FROZEN;

    // Propagate frame and compute surface vector
    propagateFrame(vertices, cur, seedIdx, originMesh);

    // Track max distance for auto-radius computation
    radius = std::max(radius, cur.distance);

    // Update neighbors (only within selected vertices)
    auto curVh = cur.vh;
    glm::vec3 curPos = Utils::toGlm(originMesh.point(curVh));

    for (const MyMesh::HalfedgeHandle &voh : originMesh.voh_range(curVh)) {
      auto nvh = originMesh.to_vertex_handle(voh);
      if (!selectedVertices.count(nvh.idx()))
        continue; // Skip non-selected

      auto &nv = vertices[nvh.idx()];

      if (nv.state == ExpMapVertex::State::FROZEN) {
        continue;
      }

      float edgeDist = glm::length(Utils::toGlm(originMesh.point(nvh)) - curPos);
      float newDist = cur.distance + edgeDist;

      if (newDist < nv.distance) {
        nv.distance = newDist;
        nv.nearest = curVh;
        nv.state = ExpMapVertex::State::ACTIVE;
        pq.push(nvh.idx());
      }
    }
  }

  // Copy UVs to mesh with normalization
  // Ensure radius is valid (avoid division by zero)
  if (radius < 1e-6f) {
    radius = 1.0f;
  }

  // Map radius to [0,1] using inscribed square: scale = 1/(radius * sqrt(2))
  float uvScale = 1.0f / (radius * std::sqrt(2.0f));

  // put the texture coords back to Mesh
  {
    // check whether Mesh has vertex texcoord2D
    if (!originMesh.has_vertex_texcoords2D()) {

      originMesh.request_vertex_texcoords2D();
      for (MyMesh::VertexHandle vh : originMesh.vertices())
        originMesh.set_texcoord2D(vh, {-1, -1});
    }

    // check whether Mesh has face texture index
    if (!originMesh.has_face_texture_index()) {

      originMesh.request_face_texture_index();
      for (MyMesh::FaceHandle fh : originMesh.faces())
        originMesh.set_texture_index(fh, -1);
    }

    for (auto vh : originMesh.vertices()) {
      auto &v = vertices[vh.idx()];
      if (v.state == ExpMapVertex::State::FROZEN) {
        // Transform geodesic coords to [0,1] UV space
        glm::vec2 uv = v.surfaceVector * uvScale + 0.5f;
        originMesh.set_texcoord2D(vh, {uv.x, uv.y});

        // get tangent / bitangent from frame
        glm::vec3 t = v.frame.axes[0];
        glm::vec3 b = v.frame.axes[1];
        originMesh.data(vh).tangent = {t.x, t.y, t.z};
        originMesh.data(vh).bitangent = {b.x, b.y, b.z};
      }
    }
  }
}

} // namespace ExpMap
