#include "hit_test.hpp"

#include <algorithm>

#include "../utils.hpp"

namespace BVH {

void BVH::build(const MyMesh &mesh) {

  _tris.clear();
  _tris.reserve(mesh.n_faces());
  for (const MyMesh::FaceHandle &fh : mesh.faces()) {

    // get triangle vertex handle
    auto fv_it = mesh.cfv_iter(fh);
    MyMesh::VertexHandle vh0 = *fv_it++;
    MyMesh::VertexHandle vh1 = *fv_it++;
    MyMesh::VertexHandle vh2 = *fv_it;

    // get point of vertices
    _tris.emplace_back(Utils::toGlm(mesh.point(vh0)), Utils::toGlm(mesh.point(vh1)),
                       Utils::toGlm(mesh.point(vh2)), fh.idx());
  }

  _nodes.clear();
  _nodes.reserve(_tris.size() * 2);

  buildRecursive(0, static_cast<int>(_tris.size()));
}

// NOLINTNEXTLINE(misc-no-recursion)
int BVH::buildRecursive(int start, int count, int depth) {

  int nodeIdx = static_cast<int>(_nodes.size());
  _nodes.push_back({});

  // calculate AABB (use index to store, not reference to avoid realloc invalidate)
  AABB aabb;
  for (int i = start; i < start + count; i++) {
    aabb.expand(_tris[i].v0);
    aabb.expand(_tris[i].v1);
    aabb.expand(_tris[i].v2);
  }
  _nodes[nodeIdx].aabb = aabb;

  if (count <= 4 || depth >= 32) {
    _nodes[nodeIdx].triStart = start;
    _nodes[nodeIdx].triCount = count;
    return nodeIdx;
  }

  // choose longest axis
  glm::vec3 extent = aabb.max - aabb.min;
  int axis = (extent.y > extent.x) ? 1 : 0;
  if (extent.z > extent[axis])
    axis = 2;

  int mid = start + count / 2;

  std::nth_element( // quicker than std::sort
      _tris.begin() + start, _tris.begin() + mid, _tris.begin() + start + count,
      [axis](const Triangle &a, const Triangle &b) {
        return (a.v0[axis] + a.v1[axis] + a.v2[axis]) < (b.v0[axis] + b.v1[axis] + b.v2[axis]);
      });

  int leftIdx = buildRecursive(start, mid - start, depth + 1);
  int rightIdx = buildRecursive(mid, start + count - mid, depth + 1);

  // access by index after realloc
  _nodes[nodeIdx].leftChild = leftIdx;
  _nodes[nodeIdx].rightChild = rightIdx;
  return nodeIdx;
}

bool BVH::intersectTriangle(const Triangle &tri, const glm::vec3 &origin, const glm::vec3 &dir, float &t) {

  glm::vec3 e1 = tri.v1 - tri.v0;
  glm::vec3 e2 = tri.v2 - tri.v0;
  glm::vec3 h = glm::cross(dir, e2);
  float a = glm::dot(e1, h);
  if (std::abs(a) < 1e-6f)
    return false;

  float f = 1.0f / a;
  glm::vec3 s = origin - tri.v0;
  float u = f * glm::dot(s, h);
  if (u < 0.0f || u > 1.0f)
    return false;

  glm::vec3 q = glm::cross(s, e1);
  float v = f * glm::dot(dir, q);
  if (v < 0.0f || u + v > 1.0f)
    return false;

  t = f * glm::dot(e2, q);
  return t > 1e-4f;
}

[[nodiscard]] HitResult BVH::raycast(const glm::vec3 &origin, const glm::vec3 &dir) const {

  HitResult result;
  if (_nodes.empty())
    return result;

  glm::vec3 invDir = 1.0f / dir; // component-wise

  // iterate DFS, use stack to avoid recursion overflow
  int stack[64];
  int stackTop = 0;
  stack[stackTop++] = 0; // root

  while (stackTop > 0) {
    int idx = stack[--stackTop];
    const BVHNode &node = _nodes[idx];

    // AABB 測試
    float tMin, tMax;
    if (!node.aabb.intersect(origin, invDir, tMin, tMax))
      continue;
    if (tMin > result.t) // farer than known nearest hit, prune
      continue;

    if (node.isLeaf()) {
      // test all triangles
      for (int i = node.triStart; i < node.triStart + node.triCount; i++) {
        float t;
        if (intersectTriangle(_tris[i], origin, dir, t)) {
          if (t < result.t) {
            result.t = t;
            result.faceIdx = _tris[i].faceIdx;
            result.hitPoint = origin + dir * t;
          }
        }
      }
    } else {
      // push far node first, so that closer node will be processed first (faster pruning)
      stack[stackTop++] = node.leftChild;
      stack[stackTop++] = node.rightChild;
    }
  }

  return result;
}

}; // namespace BVH
