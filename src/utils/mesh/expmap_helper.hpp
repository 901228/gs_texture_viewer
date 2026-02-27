#ifndef EXPMAP_HELPER_HPP
#define EXPMAP_HELPER_HPP
#pragma once

#include <set>

#include "mesh.hpp"

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

#include "../utils.hpp"

namespace ExpHelper {

inline int computeCenterFromFaces(const std::set<unsigned int> &selectedID, const MyMesh &mesh) {
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
  std::set<int> checkedVertices;
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

} // namespace ExpHelper

#endif // !EXPMAP_HELPER_HPP
