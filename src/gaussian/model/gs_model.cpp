#include "gs_model.hpp"

#include <glm/gtc/type_ptr.hpp>

#include "../utils/camera/camera.hpp"
#include "gs_gl_data.hpp"
#include "imgui.h"
#include "ply.hpp"
#include "utils.hpp"

#include "rasterizer/rasterizer.hpp"

GaussianModel::GaussianModel(int sh_degree, int device) : _sh_degree(sh_degree) {

  _initCuda(device);

  _geomBufferFunc = Utils::resizeFunctional(&_geomPtr, _allocdGeom);
  _binningBufferFunc = Utils::resizeFunctional(&_binningPtr, _allocdBinning);
  _imgBufferFunc = Utils::resizeFunctional(&_imgPtr, _allocdImg);
}

GaussianModel::GaussianModel(const char *plyPath, int sh_degree, int device)
    : GaussianModel(sh_degree, device) {

  _loadPly(plyPath);
}

void GaussianModel::_initCuda(int device) {

  DEBUG("GaussianModel Initialization");

  // initialize cuda
  {
    int num_devices;
    CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
    if (device >= num_devices) {
      if (num_devices == 0)
        ERROR("No CUDA devices detected!");
      else
        ERROR("Provided device index exceeds number of available "
              "CUDA devices!");
    }
    CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 7) {
      ERROR("Sorry, need at least compute capability 7.0+!");
    }
  }
  INFO("Initialize CUDA finished, device {}", device);
}

GaussianModel::~GaussianModel() {

  // gaussian ply data
  cudaFree(_pos_cuda);
  cudaFree(_rot_cuda);
  cudaFree(_scale_cuda);
  cudaFree(_opacity_cuda);
  cudaFree(_shs_cuda);

  // rendering data
  cudaFree(_view_cuda);
  cudaFree(_proj_cuda);
  cudaFree(_cam_pos_cuda);
  cudaFree(_background_cuda);
  cudaFree(_rect_cuda);

  if (_geomPtr)
    cudaFree(_geomPtr);
  if (_binningPtr)
    cudaFree(_binningPtr);
  if (_imgPtr)
    cudaFree(_imgPtr);
}

void GaussianModel::_loadPly(const char *plyPath) {

  // load ply
  // TODO: get degree from ply
  std::vector<Pos> pos;
  std::vector<Rot> rot;
  std::vector<Scale> scale;
  std::vector<float> opacity;
  std::vector<SHs<3>> shs;
  if (_sh_degree == 0) {
    gsCount = loadPly<0>(plyPath, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
  } else if (_sh_degree == 1) {
    gsCount = loadPly<1>(plyPath, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
  } else if (_sh_degree == 2) {
    gsCount = loadPly<2>(plyPath, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
  } else if (_sh_degree == 3) {
    gsCount = loadPly<3>(plyPath, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
  } else {
    ERROR("Unknown spherical harmonics degree: {}", _sh_degree);
  }

  _boxmin = _scenemin;
  _boxmax = _scenemax;

  int P = gsCount;

  // Allocate and fill the GPU data
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_pos_cuda, sizeof(Pos) * P));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_rot_cuda, sizeof(Rot) * P));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_shs_cuda, sizeof(SHs<3>) * P));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_opacity_cuda, sizeof(float) * P));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_scale_cuda, sizeof(Scale) * P));
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));

  // Create space for view parameters
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_view_cuda, sizeof(glm::mat4)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_proj_cuda, sizeof(glm::mat4)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_cam_pos_cuda, 3 * sizeof(float)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_background_cuda, 3 * sizeof(float)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&_rect_cuda, 2 * P * sizeof(int)));

  _gsGLData = std::make_unique<GaussianGLData>(P, (float *)pos.data(), (float *)rot.data(),
                                               (float *)scale.data(), opacity.data(), (float *)shs.data());
}

void GaussianModel::render(const Camera &camera, const int &width, const int &height,
                           const glm::vec3 &clearColor, float *image_cuda) const {

  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(_background_cuda, glm::value_ptr(clearColor), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  // Convert view and projection to target coordinate system
  glm::mat4 view_mat{camera.viewMatrix()};
  glm::mat4 proj_view_mat = camera.projectionMatrix() * view_mat;
  Utils::flipRow(view_mat, 1);
  Utils::flipRow(view_mat, 2);
  Utils::flipRow(proj_view_mat, 1);

  // Compute additional view parameters
  float tan_fovy = std::tan(camera.fov() * 0.5f);
  float tan_fovx = tan_fovy * camera.aspect();

  // Copy frame-dependent data to GPU
  CUDA_SAFE_CALL(cudaMemcpy(_view_cuda, glm::value_ptr(view_mat), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(_proj_cuda, glm::value_ptr(proj_view_mat), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(_cam_pos_cuda, glm::value_ptr(camera.eye()), sizeof(glm::vec3), cudaMemcpyHostToDevice));

  // Rasterize
  int *rects = _fastCulling ? _rect_cuda : nullptr;
  float *boxmin = _cropping ? (float *)&_boxmin : nullptr;
  float *boxmax = _cropping ? (float *)&_boxmax : nullptr;
  CUDA_SAFE_CALL(CudaRasterizer::forward(
      _geomBufferFunc, _binningBufferFunc, _imgBufferFunc, gsCount, _sh_degree, MAX_SH_COEFF,
      _background_cuda, width, height, _pos_cuda, _shs_cuda, nullptr, _opacity_cuda, _scale_cuda,
      _scalingModifier, _rot_cuda, nullptr, _view_cuda, _proj_cuda, _cam_pos_cuda, tan_fovx, tan_fovy, false,
      image_cuda, _antialiasing, nullptr, rects, boxmin, boxmax));

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error(std::format("A CUDA error occurred during rendering:{}. Please rerun "
                                         "in Debug to find the exact line!",
                                         cudaGetErrorString(cudaGetLastError())));
  }
}

void GaussianModel::controls() {

  if (ImGui::CollapsingHeader("Splat Render Option")) {
    ImGui::Indent();

    ImGui::Checkbox("Fast Culling", &_fastCulling);
    ImGui::Checkbox("Antialiasing", &_antialiasing);
    ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);

    ImGui::Checkbox("Crop Box", &_cropping);
    if (_cropping) {
      ImGui::SliderFloat("Box Min X", &_boxmin.x, _scenemin.x, _scenemax.x);
      ImGui::SliderFloat("Box Min Y", &_boxmin.y, _scenemin.y, _scenemax.y);
      ImGui::SliderFloat("Box Min Z", &_boxmin.z, _scenemin.z, _scenemax.z);
      ImGui::SliderFloat("Box Max X", &_boxmax.x, _scenemin.x, _scenemax.x);
      ImGui::SliderFloat("Box Max Y", &_boxmax.y, _scenemin.y, _scenemax.y);
      ImGui::SliderFloat("Box Max Z", &_boxmax.z, _scenemin.z, _scenemax.z);
    }

    ImGui::Unindent();
  }
}

glm::vec3 GaussianModel::center() const { return Utils::getModelCenter(_boxmin, _boxmax); }
GaussianGLData &GaussianModel::gaussianGLData() const { return *_gsGLData; }
int GaussianModel::count() const { return gsCount; }
