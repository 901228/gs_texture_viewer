#include "geodesic_splines.hpp"

#include <chrono>
#include <numbers>

#include <Eigen/Sparse>
#include <random>
#include <unordered_set>
#include <utility>

#include "../utils.hpp"
#include "hit_test.hpp"

namespace {

struct Timer {
  std::string name;
  std::chrono::high_resolution_clock::time_point start;

  Timer(const std::string &name) : name(name) { start = std::chrono::high_resolution_clock::now(); }

  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end - start).count();
    INFO("[Timer] {}: {} ms", name.c_str(), ms);
  }
};

} // namespace

namespace Mesh2Implicit {

// π(x) — project onto surface
const glm::vec3 const project(const ClosestPointResult &r) { return r.point; }

// give BVH::ClosestPointResult to avoid querying again
const glm::vec3 const interpolateNormal(const ClosestPointResult &r, const MyMesh &mesh) {
  auto fh = mesh.face_handle(r.faceIdx);
  auto fv = mesh.cfv_iter(fh);
  glm::vec3 na = Utils::toGlm(mesh.normal(*fv));
  ++fv;
  glm::vec3 nb = Utils::toGlm(mesh.normal(*fv));
  ++fv;
  glm::vec3 nc = Utils::toGlm(mesh.normal(*fv));
  return glm::normalize(Utils::barycentric(r.bary, na, nb, nc));
}

// n(x) — interpolated normal at projected point
const glm::vec3 const normal(const ClosestPointResult &r, const MyMesh &mesh) {
  return interpolateNormal(r, mesh);
}

// f(x) — signed distance to surface
// use unsigned closest-point distance & normal direction to determine sign
const float f(const glm::vec3 &x, const ClosestPointResult &r, const MyMesh &mesh) {
  glm::vec3 n = interpolateNormal(r, mesh);
  glm::vec3 diff = x - r.point;
  float sign = (glm::dot(diff, n) >= 0) ? 1.f : -1.f;
  return sign * glm::length(diff);
}

} // namespace Mesh2Implicit

namespace RadialTracing {

struct TracedPoint {
  glm::vec3 pos;     // 3D position on mesh
  glm::vec3 tangent; // parallel transported tangent (unit, in tangent plane)
  glm::vec3 normal;  // interpolated normal at pos
};

const glm::vec3 const surfaceNormal(const glm::vec3 &x, const BVH::BVH &bvh, const MyMesh &mesh) {
  auto r = bvh.closestPoint(x);
  return Mesh2Implicit::interpolateNormal(r, mesh);
}

// smallest rotation between tangent planes
const glm::vec3 const parallelTransport(const glm::vec3 &t, const glm::vec3 &n_from, const glm::vec3 &n_to) {

  // smallest rotation: rotate around the axis (n_from x n_to) by the angle (acos(n_from · n_to))
  float cosA = glm::clamp(glm::dot(n_from, n_to), -1.f, 1.f);
  if (cosA > 1.f - 1e-8f)
    return t;

  glm::vec3 axis = glm::normalize(glm::cross(n_from, n_to));
  float sinA = std::sqrt(1.f - cosA * cosA);
  // Rodrigues
  glm::vec3 result = t * cosA + glm::cross(axis, t) * sinA + axis * glm::dot(axis, t) * (1.f - cosA);
  // Re-project onto new tangent plane
  result -= glm::dot(result, n_to) * n_to;
  return glm::normalize(result);
}

const std::pair<TracedPoint, int> const substeppedProject(const TracedPoint &tp_start, float h,
                                                          const BVH::BVH &bvh, const MyMesh &mesh) {

  const float s = 1.f / std::sqrt(2.f); // cos(45°)
  const float min_step = 1e-6f;

  glm::vec3 q = tp_start.pos;
  glm::vec3 t = tp_start.tangent;
  glm::vec3 n = tp_start.normal;
  int faceIdx = -1;
  float remaining = h;

  while (remaining > min_step) {

    // Check if full remaining step is OK
    float ell = remaining;
    auto r_full = bvh.closestPoint(q + remaining * t);
    glm::vec3 n_full = Mesh2Implicit::interpolateNormal(r_full, mesh);

    if (glm::dot(n, n_full) < s) {
      // Bisect to find safe ell
      float lo = 0.f, hi = remaining;
      for (int iter = 0; iter < 32; ++iter) {
        float mid = 0.5f * (lo + hi);
        auto r_mid = bvh.closestPoint(q + mid * t);
        glm::vec3 n_mid = Mesh2Implicit::interpolateNormal(r_mid, mesh);
        if (glm::dot(n, n_mid) >= s)
          lo = mid;
        else
          hi = mid;
      }
      ell = lo;
    }

    auto r_next = bvh.closestPoint(q + ell * t);
    glm::vec3 q_next = r_next.point;
    glm::vec3 n_next = Mesh2Implicit::interpolateNormal(r_next, mesh);

    float step_dist = glm::distance(q, q_next);
    if (step_dist < min_step)
      break; // to avoid infinite loop

    t = parallelTransport(t, n, n_next);
    q = q_next;
    n = n_next;
    faceIdx = r_next.faceIdx;
    remaining -= step_dist;
  }

  return std::make_pair(TracedPoint{q, t, n}, faceIdx);
}

} // namespace RadialTracing

namespace HolonomySmoothing {

// TODO: read these 2 function & add reference of paper equations
const std::vector<float> const
computePhi(int j, const std::vector<std::vector<RadialTracing::TracedPoint>> &Q, int m) {

  std::vector<float> phi(m);
  for (int i = 0; i < m; ++i) {
    int i_next = (i + 1) % m;
    glm::vec3 t_i = Q[i][j].tangent;
    glm::vec3 t_i1 = Q[i_next][j].tangent;
    glm::vec3 n_i = Q[i][j].normal;
    glm::vec3 n_i1 = Q[i_next][j].normal;

    // Transport t_i to Q[i+1]'s tangent plane
    glm::vec3 t_hat = RadialTracing::parallelTransport(t_i, n_i, n_i1);

    // Signed angle from t_{i+1} to t_hat, around n_{i+1}
    float cosA = glm::clamp(glm::dot(t_i1, t_hat), -1.f, 1.f);
    float sinA = glm::dot(n_i1, glm::cross(t_i1, t_hat));
    phi[i] = std::atan2(sinA, cosA);
  }
  return phi;
}

struct HolonomySolver {
private:
  Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
  int m = 0;

public:
  void init(int m_, float kappa) {
    m = m_;
    Eigen::SparseMatrix<float> L(m, m);
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(m * 3);

    float diag = 4.f + 2.f / kappa;
    for (int i = 0; i < m; ++i) {
      triplets.emplace_back(i, i, diag);
      triplets.emplace_back(i, (i - 1 + m) % m, -2.f);
      triplets.emplace_back(i, (i + 1) % m, -2.f);
    }
    L.setFromTriplets(triplets.begin(), triplets.end());

    solver.compute(L);
    assert(solver.info() == Eigen::Success);
  }

  // phi: 長度 m 的 vector，來自 computePhi()
  // 回傳 theta: 長度 m 的 rotation angles
  std::vector<float> solve(const std::vector<float> &phi) const {
    assert((int)phi.size() == m);

    // RHS: Φ_j (see Eq. 11)
    // RHS of Eq. 11: rhs[i] = -2*phi[i] + 2*phi[(i-1+m)%m]
    Eigen::VectorXf rhs(m);
    for (int i = 0; i < m; ++i)
      rhs[i] = -2.f * phi[i] + 2.f * phi[(i - 1 + m) % m];

    Eigen::VectorXf theta = solver.solve(rhs);
    assert(solver.info() == Eigen::Success);

    return std::vector<float>(theta.data(), theta.data() + m);
  }
};

} // namespace HolonomySmoothing

namespace MapInterpolation {

// 對 isoline j 的 m 個 3D 點建立 periodic cubic spline
// 然後在任意角度 theta 求值
struct PeriodicSpline {
  std::vector<glm::vec3> pts; // m points
  std::vector<glm::vec3> M;   // second derivatives (moments)
  float h;                    // spacing = 2π/m

  void build(const std::vector<glm::vec3> &points) {
    pts = points;
    int m = pts.size();
    h = 2.f * std::numbers::pi_v<float> / m;

    // RHS: d_i = 6*(p_{i-1} - 2*p_i + p_{i+1}) / h²
    std::vector<glm::vec3> d(m);
    for (int i = 0; i < m; ++i)
      d[i] = 6.f / (h * h) * (pts[(i - 1 + m) % m] - 2.f * pts[i] + pts[(i + 1) % m]);

    // Solve circulant tridiagonal: [4 1 0...1; 1 4 1...0; ...] * M = d
    // 使用 Eigen（per-component，或直接對 vec3）
    using SpMat = Eigen::SparseMatrix<float>;
    SpMat A(m, m);
    std::vector<Eigen::Triplet<float>> tri;
    for (int i = 0; i < m; ++i) {
      tri.emplace_back(i, i, 4.f);
      tri.emplace_back(i, (i - 1 + m) % m, 1.f);
      tri.emplace_back(i, (i + 1) % m, 1.f);
    }
    A.setFromTriplets(tri.begin(), tri.end());
    Eigen::SparseLU<SpMat> solver;
    solver.compute(A);

    M.resize(m);
    for (int c = 0; c < 3; ++c) {
      Eigen::VectorXf rhs(m);
      for (int i = 0; i < m; ++i)
        rhs[i] = d[i][c];
      Eigen::VectorXf sol = solver.solve(rhs);
      for (int i = 0; i < m; ++i)
        M[i][c] = sol[i];
    }
  }

  glm::vec3 eval(float theta) const {
    int m = pts.size();
    // 把 theta 映射到 [0, 2π)
    theta = std::fmod(theta, 2.f * std::numbers::pi_v<float>);
    if (theta < 0)
      theta += 2.f * std::numbers::pi_v<float>;

    // 找所在 segment
    float idx_f = theta / h;
    int i = static_cast<int>(idx_f) % m;
    float t = idx_f - std::floor(idx_f); // ∈ [0,1)

    // S(t) = p_i*(1-t) + p_{i+1}*t + h²/6 * [M_i*((1-t)³-(1-t)) + M_{i+1}*(t³-t)]
    int i1 = (i + 1) % m;
    float h2_6 = h * h / 6.f;
    float ci = (1 - t) * (1 - t) * (1 - t) - (1 - t);
    float ci1 = t * t * t - t;
    return pts[i] * (1 - t) + pts[i1] * t + h2_6 * (M[i] * ci + M[i1] * ci1);
  }
};

struct NaturalSpline {
  std::vector<glm::vec3> pts;
  std::vector<float> rs;    // r values: -R, -(n-1)h, ..., 0, ..., R
  std::vector<glm::vec3> M; // moments

  void build(const std::vector<glm::vec3> &points, const std::vector<float> &rvals) {
    pts = points;
    rs = rvals;
    int N = pts.size();
    M.resize(N, glm::vec3(0));

    // Natural BC: M[0] = M[N-1] = 0
    // Tridiagonal system for interior nodes
    // h_i = rs[i+1] - rs[i] (均勻時都是 h)
    if (N < 3)
      return;

    // Thomas algorithm（均勻間距版本）
    float h = rs[1] - rs[0];
    std::vector<glm::vec3> d(N - 2);
    for (int i = 1; i < N - 1; ++i)
      d[i - 1] = 6.f / (h * h) * (pts[i - 1] - 2.f * pts[i] + pts[i + 1]);

    // 解 tridiagonal [4,1,...;1,4,1,...;...;...,1,4]
    // Thomas algorithm
    std::vector<float> c(N - 2, 1.f / 4.f); // upper diagonal / pivot
    std::vector<glm::vec3> dd(N - 2);
    dd[0] = d[0] / 4.f;
    for (int i = 1; i < N - 2; ++i) {
      float w = 1.f / (4.f - c[i - 1]);
      c[i] = w;
      dd[i] = (d[i] - dd[i - 1]) * w;
    }
    M[N - 2] = dd[N - 3];
    for (int i = N - 4; i >= 0; --i)
      M[i + 1] = dd[i] - c[i] * M[i + 2];
  }

  glm::vec3 eval(float r) const {
    int N = pts.size();
    // clamp to range
    r = glm::clamp(r, rs.front(), rs.back());

    // find segment
    float h = rs[1] - rs[0];
    int i = static_cast<int>((r - rs[0]) / h);
    i = glm::clamp(i, 0, N - 2);
    float t = (r - rs[i]) / h;

    float h2_6 = h * h / 6.f;
    float ci = (1 - t) * (1 - t) * (1 - t) - (1 - t);
    float ci1 = t * t * t - t;
    return pts[i] * (1 - t) + pts[i + 1] * t + h2_6 * (M[i] * ci + M[i + 1] * ci1);
  }
};

// q_p(r, theta): tangent disc → 3D surface point
glm::vec3 forwardMap(float r, float theta, const std::vector<PeriodicSpline> &isolineSplines, int n, float h,
                     const glm::vec3 &origin) {
  float R = n * h;
  r = glm::clamp(r, 0.f, R);

  // 對每條 isoline 在 theta 和 theta+π 各取一個點
  // 得到 2n+1 個點構成 radial curve γ̃_θ
  std::vector<glm::vec3> radialPts(2 * n + 1);
  std::vector<float> radialR(2 * n + 1);

  radialPts[n] = origin; // j=0 就是 origin
  radialR[n] = 0.f;

  for (int j = 1; j <= n; ++j) {
    radialPts[n + j] = isolineSplines[j].eval(theta);
    radialPts[n - j] = isolineSplines[j].eval(theta + std::numbers::pi_v<float>);
    radialR[n + j] = j * h;
    radialR[n - j] = -j * h;
  }

  NaturalSpline radialSpline;
  radialSpline.build(radialPts, radialR);
  return radialSpline.eval(r);
}

} // namespace MapInterpolation

namespace LogarithmicMap {

struct LogMapTable {
  std::vector<glm::vec2> uvs;   // tangent space coords
  std::vector<glm::vec3> pts3d; // corresponding 3D positions

  void build(const std::vector<MapInterpolation::PeriodicSpline> &isolineSplines, int n, float h,
             const glm::vec3 &origin, int numSamples = 5000) {

    float R = n * h;
    uvs.clear();
    pts3d.clear();
    uvs.reserve(numSamples);
    pts3d.reserve(numSamples);

    // Origin
    uvs.push_back({0, 0});
    pts3d.push_back(origin);

    // 均勻取樣 tangent disc
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distR(0, R);
    std::uniform_real_distribution<float> distTheta(0, 2 * std::numbers::pi_v<float>);

    for (int k = 0; k < numSamples; ++k) {
      float r = distR(rng);
      float theta = distTheta(rng);
      glm::vec2 uv = {r * std::cos(theta), r * std::sin(theta)};
      glm::vec3 pt = MapInterpolation::forwardMap(r, theta, isolineSplines, n, h, origin);
      uvs.push_back(uv);
      pts3d.push_back(pt);
    }
  }

  // 給定 3D 點，查詢對應 UV（brute force，可之後換 BVH）
  glm::vec2 query(const glm::vec3 &p) const {
    float best = 1e18f;
    int bestIdx = 0;
    for (int k = 0; k < (int)pts3d.size(); ++k) {
      float d2 = glm::dot(p - pts3d[k], p - pts3d[k]);
      if (d2 < best) {
        best = d2;
        bestIdx = k;
      }
    }
    return uvs[bestIdx];
  }
};

} // namespace LogarithmicMap

namespace GeodesicSplines {

void Solve(const std::unordered_set<unsigned int> &selectedID, MyMesh &originMesh, const BVH::BVH &bvh) {

  debugStruct.h = settings.h;

  // =========================================
  // Pre-Processing
  // =========================================
  glm::vec3 p(0);
  glm::vec3 e1, e2;
  {
    Timer t("Pre-Processing");

    // 1a. Calculate the center of selected faces as the origin
    int count = 0;
    for (unsigned int f : selectedID) {
      MyMesh::FaceHandle fh = originMesh.face_handle(f);
      for (const MyMesh::VertexHandle &fv : originMesh.fv_range(fh)) {
        p += Utils::toGlm(originMesh.point(fv));
        count++;
      }
    }
    p /= float(count);

    // 1b. Project p to the surface of the mesh (the center is not necessarily on the surface)
    auto r = bvh.closestPoint(p);
    p = r.point;
    debugStruct.center = p;

    // 1c. construct tangent frame (e1, e2) at p
    glm::vec3 n0 = Mesh2Implicit::interpolateNormal(r, originMesh);
    glm::vec3 arb = (std::abs(n0.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    e1 = glm::normalize(arb - glm::dot(arb, n0) * n0);
    e2 = glm::cross(n0, e1);
  }

  // =========================================
  // Section 3.1 - Radial Tracing
  // =========================================
  int N = settings.n;
  // Q[i][j] = radial curve i, step j
  std::vector<std::vector<RadialTracing::TracedPoint>> Q(settings.m,
                                                         std::vector<RadialTracing::TracedPoint>(N + 1));
  {
    Timer t("Radial Tracing");

    debugStruct.Q.clear();
    debugStruct.T.clear();
    for (int i = 0; i < settings.m; ++i) {
      float angle = 2.f * std::numbers::pi_v<float> * i / settings.m;
      glm::vec3 t = std::cos(angle) * e1 + std::sin(angle) * e2;
      auto r0 = bvh.closestPoint(p);
      glm::vec3 n0 = Mesh2Implicit::interpolateNormal(r0, originMesh);
      Q[i][0] = {p, t, n0};
      debugStruct.Q.push_back({});
      debugStruct.T.push_back({});
    }

    HolonomySmoothing::HolonomySolver holonomySolver;
    if (settings.enableSmoothing) {
      holonomySolver.init(settings.m, settings.kappa);
    }

    // tracing loop
    for (int j = 0; j < N; ++j) {

      bool dirty = false;
      for (int i = 0; i < settings.m; i++) {

        const RadialTracing::TracedPoint &tp_cur = Q[i][j];
        RadialTracing::TracedPoint tp_next;
        int faceIdx = -1;
        if (!settings.useSubSteppedProject) {
          // Eq. 2: q_{i,j+1} = π(q_{i,j} + h * t_{i,j})
          auto r = bvh.closestPoint(tp_cur.pos + settings.h * tp_cur.tangent);
          tp_next.pos = r.point;
          faceIdx = r.faceIdx;

          tp_next.normal = RadialTracing::surfaceNormal(tp_next.pos, bvh, originMesh);
          tp_next.tangent = RadialTracing::parallelTransport(tp_cur.tangent, tp_cur.normal, tp_next.normal);
        } else {

          // =========================================
          // Section 3.2 - Substepping
          // =========================================

          // Eq. 3: n(q_{i}, j) · n(π(τ_{i,j}(ℓ))) = s   (ℓ ∈ [0, h])
          std::tie(tp_next, faceIdx) = RadialTracing::substeppedProject(tp_cur, settings.h, bvh, originMesh);
        }

        if (faceIdx == -1 || selectedID.contains(faceIdx)) {
          dirty = true;
        }

        Q[i][j + 1] = {tp_next.pos, tp_next.tangent, tp_next.normal};
        debugStruct.Q[i].push_back(tp_next.pos);
        debugStruct.T[i].push_back(tp_next.tangent);
      }

      if (!dirty) {
        N = j;
        for (int i = 0; i < settings.m; i++) {
          Q[i][j + 1] = {};
          debugStruct.Q[i].pop_back();
          debugStruct.T[i].pop_back();
        }
        break;
      }

      // =========================================
      // Section 3.3 - Holonomy Smoothing
      // =========================================
      if (settings.enableSmoothing) {

        auto phi = HolonomySmoothing::computePhi(j + 1, Q, settings.m);
        auto theta = holonomySolver.solve(phi);

        for (int i = 0; i < settings.m; ++i) {
          glm::vec3 n = Q[i][j + 1].normal;
          glm::vec3 &t = Q[i][j + 1].tangent;
          float a = theta[i];
          t = t * std::cos(a) + glm::cross(n, t) * std::sin(a);
          t -= glm::dot(t, n) * n;
          t = glm::normalize(t);
        }

        // Debug: cache phi & theta for display
        debugStruct.phi.push_back(phi);
        debugStruct.theta.push_back(theta);
      }
    }
  }

  // =========================================
  // Section 3.4 - Map Interpolation
  // =========================================
  std::vector<MapInterpolation::PeriodicSpline> isolineSplines(N + 1);
  {
    Timer t("Map Interpolation");

    for (int j = 0; j <= N; ++j) {
      std::vector<glm::vec3> isoPts(settings.m);
      for (int i = 0; i < settings.m; ++i)
        isoPts[i] = Q[i][j].pos;
      isolineSplines[j].build(isoPts);
    }
  }

  // =========================================
  // Section 3.5 - Logarithmic Map
  // =========================================
  // 建立 log map table（tracing 結束後執行一次）
  LogarithmicMap::LogMapTable logMap;
  {
    Timer t("Logarithmic Map");

    logMap.build(isolineSplines, N, settings.h, p);
  }

  {
    Timer t("Write Texture Coordinates");

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

    // 寫入 UV
    originMesh.request_halfedge_texcoords2D();
    for (auto vh : originMesh.vertices()) {
      glm::vec3 vpos = Utils::toGlm(originMesh.point(vh));
      glm::vec2 uv = logMap.query(vpos);

      // 正規化到 [0, 1]（OpenMesh UV 慣例）
      float R = N * settings.h;
      uv = uv / (2.f * R) + glm::vec2(0.5f);
      // uv = glm::clamp(uv, 0.f, 1.f);

      originMesh.set_texcoord2D(vh, {uv.x, uv.y});
    }
  }
}

} // namespace GeodesicSplines
