#include "texture.hpp"

#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#include <glad/gl.h>

#include <ImGui/imgui.h>

#include <stb_image.h>
#include <toml++/toml.hpp>

#include "../gl/program.hpp"
#include "../logger.hpp"
#include "utils/utils.hpp"

int TextureWrap::gl(Mode mode) {
  switch (mode) {
  case Mode::Repeat:
    return GL_REPEAT;
  case Mode::Mirror:
    return GL_MIRRORED_REPEAT;
  case Mode::Clamp:
    return GL_CLAMP_TO_EDGE;
  case Mode::Border:
    return GL_CLAMP_TO_BORDER;
  default:
    throw std::runtime_error("Unknown texture wrap mode!");
  }
}

cudaTextureAddressMode TextureWrap::cuda(Mode mode) {
  switch (mode) {
  case Mode::Repeat:
    return cudaAddressModeWrap;
  case Mode::Mirror:
    return cudaAddressModeMirror;
  case Mode::Clamp:
    return cudaAddressModeClamp;
  case Mode::Border:
    return cudaAddressModeBorder;
  default:
    throw std::runtime_error("Unknown texture wrap mode!");
  }
}

std::unique_ptr<ImageTexture> ImageTexture::create(const std::string &path, TextureWrap::Mode wrapX,
                                                   TextureWrap::Mode wrapY) {
  unsigned int id;
  float width, height;
  ColorType colorType;
  if (!loadImage(path, id, width, height, wrapX, wrapY, colorType)) {
    return nullptr;
  }

  return std::make_unique<ImageTexture>(path, id, width, height, wrapX, wrapY, colorType);
}

ImageTexture::ImageTexture(const std::string &path, const unsigned int &id, const float &width,
                           const float &height, TextureWrap::Mode wrapX, TextureWrap::Mode wrapY,
                           ColorType colorType)
    : _path(path), _name(Utils::File::stem(path)), _id(id), _width(width), _height(height), _wrapX(wrapX),
      _wrapY(wrapY), _colorType(colorType) {

  _cudaTextureId.reset();
}

ImageTexture::~ImageTexture() {
  glDeleteTextures(1, &_id);
  if (_cudaTextureId.has_value()) {
    cudaDestroyTextureObject(_cudaTextureId.value());
  }
}

bool ImageTexture::loadImage(const std::string &path, unsigned int &id, float &width, float &height,
                             TextureWrap::Mode wrapX, TextureWrap::Mode wrapY, ColorType &colorType) {

  unsigned char *data = nullptr;
  try {
    glGenTextures(1, &id);

    int w, h, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    data = stbi_load(path.c_str(), &w, &h, &nrComponents, 0);

    if (data) {

      GLint format, internalformat;
      if (nrComponents == 4) {
        colorType = ColorType::RGBA;
        format = GL_RGBA;
        internalformat = GL_RGBA;
      } else if (nrComponents == 3) {
        colorType = ColorType::RGB;
        format = GL_RGB;
        internalformat = GL_RGB;
      } else if (nrComponents == 1) {
        colorType = ColorType::R;
        format = GL_RED;
        internalformat = GL_R8;
      } else {
        throw std::runtime_error(std::format("Unsupported image with channel count: {}", nrComponents));
      }

      glBindTexture(GL_TEXTURE_2D, id);
      glTexImage2D(GL_TEXTURE_2D, 0, internalformat, w, h, 0, format, GL_UNSIGNED_BYTE, data);
      glGenerateMipmap(GL_TEXTURE_2D);

      width = static_cast<float>(w);
      height = static_cast<float>(h);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, TextureWrap::gl(wrapX));
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, TextureWrap::gl(wrapY));
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      stbi_image_free(data);
      return true;
    } else {
      throw std::runtime_error(std::format("Failed to load image file {}", path));
    }
  } catch (const std::exception &err) {
    stbi_image_free(data);
    ERROR("Failed to load texture: {}", err.what());
  }

  return false;
}

void ImageTexture::setupUniforms(const Program &program, unsigned int index, std::string location) const {

  glActiveTexture(GL_TEXTURE0 + index);
  glBindTexture(GL_TEXTURE_2D, _id);
  program.setInt((!location.empty() ? location : std::format("tex[{}]", index)).c_str(),
                 static_cast<int>(index));
}

bool ImageTexture::toCuda() {

  try {
    int w, h, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load(_path.c_str(), &w, &h, &nrComponents, 4);

    if (data) {

      // allocate CUDA memory
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
      cudaArray_t cuArray;
      cudaMallocArray(&cuArray, &channelDesc, w, h);

      // copy data to CUDA device
      const size_t spitch = w * 4 * sizeof(unsigned char); // 每行 bytes
      cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch,     // src pointer, src pitch
                          spitch, h,                       // copy width (bytes), copy height
                          cudaMemcpyHostToDevice);

      // create texture object
      cudaResourceDesc resDesc = {};
      resDesc.resType = cudaResourceTypeArray;
      resDesc.res.array.array = cuArray;

      cudaTextureDesc texDesc = {};
      texDesc.addressMode[0] = TextureWrap::cuda(_wrapX); // TEXTURE_WRAP
      texDesc.addressMode[1] = TextureWrap::cuda(_wrapY);
      texDesc.readMode = cudaReadModeNormalizedFloat; // uchar4 → float4 [0,1]
      texDesc.filterMode = cudaFilterModeLinear;      // bilinear interpolation
      // float4 pixel = tex2D<float4>(texObj, u, v);
      texDesc.normalizedCoords = 1; // use 0~1 UV coordinates

      cudaTextureObject_t texObj;
      cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
      if (err == cudaSuccess) {
        _cudaTextureId = texObj;
      } else {
        throw std::runtime_error(std::format("Failed to create texture object: {}", cudaGetErrorString(err)));
      }

      stbi_image_free(data);
      return true;
    } else {

      stbi_image_free(data);
      throw std::runtime_error(std::format("Failed to load image file {}", _path));
    }
  } catch (const std::exception &err) {
    ERROR("Failed to load texture: {}", err.what());
  }

  return false;
}

void ImageTexture::saveTextureList(const std::vector<std::unique_ptr<ImageTexture>> &list,
                                   const std::string_view &filepath) {

  toml::table tbl;
  if (std::filesystem::exists(filepath)) {
    try {
      auto result = toml::parse_file(filepath);
      tbl = std::move(result);
    } catch (const std::exception &err) {
      ERROR("Parsing failed: {}", err.what());
    }
  }

  toml::array texturesList;
  for (const auto &texture : list) {
    texturesList.push_back(texture->path());
  }

  tbl.insert_or_assign("texturesList", texturesList);

  std::ofstream file(std::string(filepath).c_str());
  file << tbl;
  file.close();
}

std::vector<std::unique_ptr<ImageTexture>> ImageTexture::loadTextureList(const std::string_view &filepath) {

  try {
    std::vector<std::unique_ptr<ImageTexture>> result;

    toml::table tbl = toml::parse_file(filepath);

    toml::array *arr = tbl["texturesList"].as_array();
    if (!arr) {
      throw std::runtime_error("texturesList is not an array");
    }

    for (const auto &elem : *arr) {
      if (std::optional<std::string> path = elem.value<std::string>(); path) {
        result.push_back(ImageTexture::create(path.value()));
      }
    }

    return result;
  } catch (const std::exception &err) {
    ERROR("Parsing failed: {}", err.what());
  }
  return {};
}

PBRTexture::PBRTexture(const std::string path, std::string basecolorPath, std::string normalPath,
                       std::string heightPath, float heightScale)
    : _path(path), _name(Utils::File::stem(path)), _heightScale(heightScale) {

  auto directoryPath = std::filesystem::path(path);
  _basecolor = ImageTexture::create((directoryPath / basecolorPath).string());
  _normal = ImageTexture::create((directoryPath / normalPath).string());
  _height = ImageTexture::create((directoryPath / heightPath).string());

  assert(_basecolor->colorType() == ImageTexture::ColorType::RGBA ||
         _basecolor->colorType() == ImageTexture::ColorType::RGB);
  assert(_normal->colorType() == ImageTexture::ColorType::RGB);
  assert(_height->colorType() == ImageTexture::ColorType::R);
}

PBRTexture::~PBRTexture() = default;

void PBRTexture::controls() { ImGui::SliderFloat("height scale", &_heightScale, 0.0f, 1.0f); }

void PBRTexture::setupUniforms(const Program &program, unsigned int index,
                               const PBRTextureLocation &location) const {

  glActiveTexture(GL_TEXTURE0 + index);
  glBindTexture(GL_TEXTURE_2D, _basecolor->id());
  program.setInt(location.basecolor.c_str(), index);

  glActiveTexture(GL_TEXTURE1 + index);
  glBindTexture(GL_TEXTURE_2D, _normal->id());
  program.setInt(location.normal.c_str(), index + 1);

  glActiveTexture(GL_TEXTURE2 + index);
  glBindTexture(GL_TEXTURE_2D, _height->id());
  program.setInt(location.height.c_str(), index + 2);

  program.setFloat(location.heightScale.c_str(), heightScale());
}

void PBRTexture::saveTextureList(const std::vector<std::unique_ptr<PBRTexture>> &list,
                                 const std::string_view &filepath) {

  toml::table tbl;
  if (std::filesystem::exists(filepath)) {
    try {
      auto result = toml::parse_file(filepath);
      tbl = std::move(result);
    } catch (const std::exception &err) {
      ERROR("Parsing failed: {}", err.what());
    }
  }

  // re-create the array
  toml::array pbrList;
  for (const auto &pbr : list) {
    toml::table entry;
    entry.insert_or_assign("path", pbr->_path);
    entry.insert_or_assign("basecolor", Utils::File::filename(pbr->_basecolor->path()));
    entry.insert_or_assign("normal", Utils::File::filename(pbr->_normal->path()));
    entry.insert_or_assign("height", Utils::File::filename(pbr->_height->path()));
    entry.insert_or_assign("heightScale", pbr->_heightScale);
    pbrList.push_back(entry);
  }

  tbl.insert_or_assign("pbrList", pbrList);

  std::ofstream file(std::string(filepath).c_str());
  file << tbl;
  file.close();
}

std::vector<std::unique_ptr<PBRTexture>> PBRTexture::loadTextureList(const std::string_view &filepath) {

  try {
    std::vector<std::unique_ptr<PBRTexture>> result;

    toml::table tbl = toml::parse_file(filepath);

    toml::array *arr = tbl["pbrList"].as_array();
    if (!arr) {
      throw std::runtime_error("pbrList is not an array");
    }

    for (const auto &elem : *arr) {
      const auto *entry = elem.as_table();
      if (!entry)
        continue;

      auto path = entry->get_as<std::string>("path");
      auto basecolor = entry->get_as<std::string>("basecolor");
      auto normal = entry->get_as<std::string>("normal");
      auto height = entry->get_as<std::string>("height");
      auto heightScale = entry->get_as<double>("heightScale");

      if (!basecolor || !normal || !height)
        continue; // continue if any of the fields is missing

      if (heightScale) {
        result.push_back(std::make_unique<PBRTexture>(
            path->get(), Utils::File::filename(basecolor->get()), Utils::File::filename(normal->get()),
            Utils::File::filename(height->get()), static_cast<float>(heightScale->get())));
      } else {
        result.push_back(std::make_unique<PBRTexture>(path->get(), Utils::File::filename(basecolor->get()),
                                                      Utils::File::filename(normal->get()),
                                                      Utils::File::filename(height->get())));
      }
    }

    return result;
  } catch (const std::exception &err) {
    ERROR("Parsing failed: {}", err.what());
  }
  return {};
}
