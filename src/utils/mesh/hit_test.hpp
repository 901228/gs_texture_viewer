#ifndef HIT_TEST_HPP
#define HIT_TEST_HPP
#pragma once

#include <glm/glm.hpp>

#include "mesh.hpp"

struct HitResult {
  glm::vec3 hitPoint{};
  glm::vec3 bary{};
  float t = 1e9f;
  int faceIdx = -1;
};

/**
 * For Geodesic Splines
 * Given a point, find the closest point on the mesh.
 */
struct ClosestPointResult {
  glm::vec3 point{};
  glm::vec3 bary{}; // barycentric (u, v, w), u+v+w=1
  float dist2 = 1e18f;
  int faceIdx = -1;
};

namespace BVH {

struct AABB {
  glm::vec3 min{1e9f};
  glm::vec3 max{-1e9f};

  inline void expand(const glm::vec3 &p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }

  inline void expand(const AABB &other) {
    min = glm::min(min, other.min);
    max = glm::max(max, other.max);
  }

  [[nodiscard]] inline glm::vec3 centroid() const { return (min + max) * 0.5f; }

  // slab method
  inline bool intersect(const glm::vec3 &origin, const glm::vec3 &invDir, float &tMin, float &tMax) const {
    glm::vec3 t0 = (min - origin) * invDir;
    glm::vec3 t1 = (max - origin) * invDir;
    glm::vec3 tSmall = glm::min(t0, t1);
    glm::vec3 tLarge = glm::max(t0, t1);
    tMin = glm::max(tSmall.x, glm::max(tSmall.y, tSmall.z));
    tMax = glm::min(tLarge.x, glm::min(tLarge.y, tLarge.z));
    return tMax >= tMin && tMax > 1e-4f;
  }

  inline float minDist2FromPoint(const glm::vec3 &p) const {
    glm::vec3 clamped = glm::clamp(p, min, max);
    glm::vec3 diff = p - clamped;
    return glm::dot(diff, diff);
  }
};

struct BVHNode {
  AABB aabb;
  int leftChild = -1; // index into _nodes, -1 = leaf
  int rightChild = -1;
  int triStart = -1; // index into _tris (leaf only)
  int triCount = 0;

  [[nodiscard]] inline bool isLeaf() const { return leftChild == -1; }
};

struct Triangle {
  glm::vec3 v0, v1, v2;
  int faceIdx;
};

class BVH {
private:
  std::vector<BVHNode> _nodes;
  std::vector<Triangle> _tris; // re-order triangles while building

public:
  // construct from triangle list
  void build(const MyMesh &mesh);

  // raycast, return nearest face
  [[nodiscard]] HitResult raycast(const glm::vec3 &origin, const glm::vec3 &dir) const;

private:
  // build recursively, return node index
  int buildRecursive(int start, int count, int depth = 0);

  // Möller–Trumbore
  static bool intersectTriangle(const Triangle &tri, const glm::vec3 &origin, const glm::vec3 &dir, float &t,
                                float &u, float &v);

public:
  // closest point on mesh
  [[nodiscard]] ClosestPointResult closestPoint(const glm::vec3 &p) const;

private:
  // Eberly's point-to-triangle closest point
  static ClosestPointResult closestPointOnTriangle(const glm::vec3 &p, const Triangle &tri);
};

} // namespace BVH

#endif // !HIT_TEST_HPP
