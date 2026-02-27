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
  static std::unique_ptr<ImageTexture> create(const std::string &path,
                                              TextureWrap::Mode wrapX = TextureWrap::Mode::Repeat,
                                              TextureWrap::Mode wrapY = TextureWrap::Mode::Repeat);

  explicit ImageTexture(const std::string &path, const unsigned int &id, const float &width,
                        const float &height, TextureWrap::Mode wrapX, TextureWrap::Mode wrapY);
  ~ImageTexture();

private:
  static bool loadImage(const std::string &path, unsigned int &id, float &width, float &height,
                        TextureWrap::Mode wrapX, TextureWrap::Mode wrapY);

private:
  unsigned int _id = 0;
  std::string _path;
  std::string _name;
  float _width = 1;
  float _height = 1;

  TextureWrap::Mode _wrapX;
  TextureWrap::Mode _wrapY;

public:
  [[nodiscard]] inline unsigned int id() const { return _id; }
  [[nodiscard]] inline std::string path() const { return _path; }
  [[nodiscard]] inline std::string name() const { return _name; }
  [[nodiscard]] inline float aspect() const { return _width / _height; }

public:
  void setupUniforms(const Program &program, unsigned int index) const;

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
  static void saveTextureList(const std::vector<std::unique_ptr<ImageTexture>> &list);
  static std::vector<std::unique_ptr<ImageTexture>> loadTextureList(const std::string_view &filepath);
};

#endif // !TEXTURE_HPP
