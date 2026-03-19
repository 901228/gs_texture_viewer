#include "solve_uv.hpp"

#include "../utils.hpp"

#include "expmap.hpp"
#include "harmonics.hpp"
#include "model.hpp"
#include "utils/mesh/mesh.hpp"

namespace SolveUV {

void calculateTB(MyMesh &mesh) {

  std::vector<glm::vec3> vtangent(mesh.n_vertices(), glm::vec3(0));
  std::vector<glm::vec3> vbitangent(mesh.n_vertices(), glm::vec3(0));

  for (auto fh : mesh.faces()) {

    auto fv = mesh.cfv_iter(fh);
    auto vh0 = *fv;
    ++fv;
    auto vh1 = *fv;
    ++fv;
    auto vh2 = *fv;

    glm::vec3 p0 = Utils::toGlm(mesh.point(vh0));
    glm::vec3 p1 = Utils::toGlm(mesh.point(vh1));
    glm::vec3 p2 = Utils::toGlm(mesh.point(vh2));

    auto tc0 = mesh.texcoord2D(vh0);
    auto tc1 = mesh.texcoord2D(vh1);
    auto tc2 = mesh.texcoord2D(vh2);

    // skip face without UV
    if (tc0[0] < 0 || tc1[0] < 0 || tc2[0] < 0)
      continue;

    glm::vec2 uv0(tc0[0], tc0[1]);
    glm::vec2 uv1(tc1[0], tc1[1]);
    glm::vec2 uv2(tc2[0], tc2[1]);

    glm::vec3 dp1 = p1 - p0, dp2 = p2 - p0;
    glm::vec2 duv1 = uv1 - uv0, duv2 = uv2 - uv0;

    float denom = duv1.x * duv2.y - duv2.x * duv1.y;
    if (std::abs(denom) < 1e-8f)
      continue;

    float inv = 1.f / denom;
    glm::vec3 T = inv * (duv2.y * dp1 - duv1.y * dp2);
    glm::vec3 B = inv * (-duv2.x * dp1 + duv1.x * dp2);

    // Accumulate (area-weighted 可以更精確，但這樣已夠)
    vtangent[vh0.idx()] += T;
    vtangent[vh1.idx()] += T;
    vtangent[vh2.idx()] += T;
    vbitangent[vh0.idx()] += B;
    vbitangent[vh1.idx()] += B;
    vbitangent[vh2.idx()] += B;
  }

  // notmalzie and write to mesh
  for (auto vh : mesh.vertices()) {
    glm::vec3 T = vtangent[vh.idx()];
    glm::vec3 B = vbitangent[vh.idx()];
    if (glm::length(T) < 1e-8f || glm::length(B) < 1e-8f)
      continue;

    glm::vec3 n = Utils::toGlm(mesh.normal(vh));

    // Gram-Schmidt orthogonalize against normal
    T = glm::normalize(T - glm::dot(T, n) * n);
    B = glm::normalize(B - glm::dot(B, n) * n - glm::dot(B, T) * T);

    mesh.data(vh).tangent = {T.x, T.y, T.z};
    mesh.data(vh).bitangent = {B.x, B.y, B.z};
  }
}

std::pair<LogarithmicMap::LogMapTable, float> SolveGeodesic(glm::vec3 hitPoint,
                                                            GeodesicSplines::Implicit &model) {
  return GeodesicSplines::Solve(hitPoint, model);
}

void Solve(const SolvingMode &mode, const std::unordered_set<unsigned int> &selectedID, Model &model,
           std::optional<glm::vec3> hitPoint) {

  switch (mode) {
  case SolvingMode::Harmonics:
    Harmonics::Solve(selectedID, model.mesh());
    break;
  case SolvingMode::ExpMap:
    ExpMap::Solve(selectedID, model.mesh());
    break;
  case SolvingMode::GeodesicSplines: {
    glm::vec3 center;
    if (hitPoint.has_value()) {
      center = hitPoint.value();
    } else {
      int count = 0;
      for (unsigned int f : selectedID) {
        MyMesh::FaceHandle fh = model.mesh().face_handle(f);
        for (const MyMesh::VertexHandle &fv : model.mesh().fv_range(fh)) {
          center += Utils::toGlm(model.mesh().point(fv));
          count++;
        }
      }
      center /= float(count);
    }
    const auto [logMap, R] = SolveGeodesic(center, model);

    {
      Utils::Timer::Timer t("Write Texture Coordinates");
      MyMesh &mesh = model.mesh();

      // check whether Mesh has vertex texcoord2D
      if (!mesh.has_vertex_texcoords2D()) {

        mesh.request_vertex_texcoords2D();
        for (MyMesh::VertexHandle vh : mesh.vertices())
          mesh.set_texcoord2D(vh, {-1, -1});
      }

      // check whether Mesh has face texture index
      if (!mesh.has_face_texture_index()) {

        mesh.request_face_texture_index();
        for (MyMesh::FaceHandle fh : mesh.faces())
          mesh.set_texture_index(fh, -1);
      }

      // write uv
      mesh.request_halfedge_texcoords2D();
      for (auto vh : mesh.vertices()) {
        glm::vec3 vpos = Utils::toGlm(mesh.point(vh));
        glm::vec2 uv = logMap.query(vpos);

        // normalize to [0, 1]
        glm::vec2 uvNorm = uv / (2.f * R) + glm::vec2(0.5f);
        mesh.set_texcoord2D(vh, {uvNorm.x, uvNorm.y});

        // // Tangent/Bitangent: finite differences on UV
        // float eps = R * 0.02f; // about 2% of map radius

        // // +u
        // glm::vec2 uv_du = uv + glm::vec2(eps, 0.f);
        // float r_du = glm::length(uv_du);
        // float theta_du = std::atan2(uv_du.y, uv_du.x);
        // glm::vec3 pt_du = MapInterpolation::forwardMap(r_du, theta_du, isolineSplines, N, settings.h, p);

        // // +v
        // glm::vec2 uv_dv = uv + glm::vec2(0.f, eps);
        // float r_dv = glm::length(uv_dv);
        // float theta_dv = std::atan2(uv_dv.y, uv_dv.x);
        // glm::vec3 pt_dv = MapInterpolation::forwardMap(r_dv, theta_dv, isolineSplines, N, settings.h, p);

        // glm::vec3 tangent = glm::normalize(pt_du - vpos);
        // glm::vec3 bitangent = glm::normalize(pt_dv - vpos);

        // originMesh.data(vh).tangent = {tangent.x, tangent.y, tangent.z};
        // originMesh.data(vh).bitangent = {bitangent.x, bitangent.y, bitangent.z};
      }

      SolveUV::calculateTB(mesh);
    }
    break;
  }
  default:
    throw std::runtime_error("Unknown solving mode!");
  }
}

} // namespace SolveUV
