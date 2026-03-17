#ifndef TEXTURE_HPP
#define TEXTURE_HPP
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

class Program;

namespace TextureWrap {

enum class Mode : int { Repeat, Mirror, Clamp, Border };

int gl(Mode mode);
cudaTextureAddressMode cuda(Mode mode);

}; // namespace TextureWrap

class ImageTexture {
public:
  enum class ColorType { Auto = 0, RGBA = 4, RGB = 3, R = 1 };

  static std::unique_ptr<ImageTexture> create(const std::string &path, ColorType colorType = ColorType::Auto,
                                              TextureWrap::Mode wrapX = TextureWrap::Mode::Repeat,
                                              TextureWrap::Mode wrapY = TextureWrap::Mode::Repeat);

  explicit ImageTexture(const std::string &path, const unsigned int &id, const float &width,
                        const float &height, TextureWrap::Mode wrapX, TextureWrap::Mode wrapY,
                        ColorType colorType);
  ~ImageTexture();

private:
  static bool loadImage(const std::string &path, unsigned int &id, float &width, float &height,
                        TextureWrap::Mode wrapX, TextureWrap::Mode wrapY, ColorType &colorType);

private:
  unsigned int _id = 0;
  std::string _path;
  std::string _name;
  float _width = 1;
  float _height = 1;

  TextureWrap::Mode _wrapX;
  TextureWrap::Mode _wrapY;

  ColorType _colorType;

public:
  [[nodiscard]] inline unsigned int id() const { return _id; }
  [[nodiscard]] inline std::string path() const { return _path; }
  [[nodiscard]] inline std::string name() const { return _name; }
  [[nodiscard]] inline float aspect() const { return _width / _height; }
  [[nodiscard]] inline ColorType colorType() const { return _colorType; }

public:
  void setupUniforms(const Program &program, unsigned int index, std::string location = {}) const;

private:
  std::optional<cudaTextureObject_t> _cudaTextureId;
  bool toCuda();

public:
  [[nodiscard]] inline cudaTextureObject_t cudaTextureId() {
    if (!_cudaTextureId.has_value()) {
      toCuda();
    }
    return _cudaTextureId.value();
  }

public:
  static void saveTextureList(const std::vector<std::unique_ptr<ImageTexture>> &list,
                              const std::string_view &filepath);
  static std::vector<std::unique_ptr<ImageTexture>> loadTextureList(const std::string_view &filepath);
};

class PBRTexture {
public:
  PBRTexture(const std::string path, std::string basecolorPath, std::string normalPath,
             std::string heightPath, float heightScale = 0.0f);
  ~PBRTexture();

private:
  std::string _name;
  std::string _path;
  std::unique_ptr<ImageTexture> _basecolor;
  std::unique_ptr<ImageTexture> _normal;
  std::unique_ptr<ImageTexture> _height;

  float _heightScale;

public:
  [[nodiscard]] inline ImageTexture &basecolor() const { return *_basecolor; }
  [[nodiscard]] inline ImageTexture &normal() const { return *_normal; }
  [[nodiscard]] inline ImageTexture &height() const { return *_height; }

  [[nodiscard]] inline float heightScale() const { return _heightScale; }

  [[nodiscard]] inline std::string name() const { return _name; }

public:
  struct PBRTextureLocation {
    std::string basecolor;
    std::string normal;
    std::string height;
    std::string heightScale;
    PBRTextureLocation(std::string basecolor = "material.basecolor", std::string normal = "material.normal",
                       std::string height = "heightMap", std::string heightScale = "heightScale")
        : basecolor(basecolor), normal(normal), height(height), heightScale(heightScale) {}
  };
  void setupUniforms(const Program &program, unsigned int index = 0,
                     const PBRTextureLocation &location = {}) const;

public:
  void controls();

public:
  static void saveTextureList(const std::vector<std::unique_ptr<PBRTexture>> &list,
                              const std::string_view &filepath);
  static std::vector<std::unique_ptr<PBRTexture>> loadTextureList(const std::string_view &filepath);
};

#endif // !TEXTURE_HPP
