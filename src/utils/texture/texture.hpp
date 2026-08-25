#ifndef TEXTURE_HPP
#define TEXTURE_HPP
#pragma once

#include <memory>
#include <string>
#include <vector>

class Program;

namespace TextureWrap {

enum class Mode : int { Repeat, Mirror, Clamp, Border };

int gl(Mode mode);

}; // namespace TextureWrap

class ImageTexture {
public:
  enum class ColorType { Auto = 0, RGBA = 4, RGB = 3, R = 1 };

  static std::unique_ptr<ImageTexture> create(const std::string &path, ColorType colorType = ColorType::Auto,
                                              TextureWrap::Mode wrapX = TextureWrap::Mode::Border,
                                              TextureWrap::Mode wrapY = TextureWrap::Mode::Border);

  // Create a texture from an in-memory encoded image buffer (e.g. a glb embedded
  // PNG/JPEG). Unlike create(), the image is NOT flipped vertically so it follows
  // the glTF UV convention (origin at top-left).
  static std::unique_ptr<ImageTexture> createFromMemory(const std::string &name, const unsigned char *buffer,
                                                        int length, ColorType colorType = ColorType::Auto,
                                                        TextureWrap::Mode wrapX = TextureWrap::Mode::Repeat,
                                                        TextureWrap::Mode wrapY = TextureWrap::Mode::Repeat);

  explicit ImageTexture(const std::string &path, const unsigned int &id, const float &width,
                        const float &height, TextureWrap::Mode wrapX, TextureWrap::Mode wrapY,
                        ColorType colorType);
  ~ImageTexture();

private:
  static bool loadImage(const std::string &path, unsigned int &id, float &width, float &height,
                        TextureWrap::Mode wrapX, TextureWrap::Mode wrapY, ColorType &colorType);
  static bool loadImageFromMemory(const unsigned char *buffer, int length, unsigned int &id, float &width,
                                  float &height, TextureWrap::Mode wrapX, TextureWrap::Mode wrapY,
                                  ColorType &colorType);
  // Uploads decoded pixels to a freshly generated GL texture. `data` holds `channels`
  // interleaved 8-bit components. Returns false (and frees nothing) on unsupported channel counts.
  static bool uploadToGL(const unsigned char *data, int w, int h, int channels, unsigned int &id,
                         float &width, float &height, TextureWrap::Mode wrapX, TextureWrap::Mode wrapY,
                         ColorType &colorType);

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

public:
  static void saveTextureList(const std::vector<std::unique_ptr<ImageTexture>> &list,
                              const std::string_view &filepath);
  static std::vector<std::unique_ptr<ImageTexture>> loadTextureList(const std::string_view &filepath);
};

class PBRTexture {
public:
  PBRTexture(const std::string path, std::string basecolorPath, std::string normalPath,
             std::string heightPath, std::string roughnessPath, std::string maskPath,
             float heightScale = 0.01f);
  ~PBRTexture();

private:
  std::string _name;
  std::string _path;
  std::unique_ptr<ImageTexture> _basecolor;
  std::unique_ptr<ImageTexture> _normal;
  std::unique_ptr<ImageTexture> _height;
  std::unique_ptr<ImageTexture> _roughness;
  std::unique_ptr<ImageTexture> _mask;

  float _heightScale;
  float _roughnessScale = 1.0f; // multiplies the sampled roughness of the decal

public:
  // How the height map is applied. The integer values must match the heightMode
  // uniform consumed by shader.tese / shader.frag.
  enum class HeightMode : int { None = 0, ParallaxOcclusion = 1, TessellationDisplacement = 2 };

private:
  HeightMode _heightMode = HeightMode::None;
  int _tessLevel = 32;          // subdivision level used in TessellationDisplacement mode
  bool _invertHeight = false;   // flip the displacement direction (e.g. when mesh normals point inward)

public:
  [[nodiscard]] inline ImageTexture &basecolor() const { return *_basecolor; }
  [[nodiscard]] inline ImageTexture &normal() const { return *_normal; }
  [[nodiscard]] inline ImageTexture &height() const { return *_height; }
  [[nodiscard]] inline ImageTexture &roughness() const { return *_roughness; }
  [[nodiscard]] inline ImageTexture &mask() const { return *_mask; }

  [[nodiscard]] inline float heightScale() const { return _heightScale; }
  [[nodiscard]] inline float roughnessScale() const { return _roughnessScale; }

  [[nodiscard]] inline std::string name() const { return _name; }

public:
  struct PBRTextureLocation {
    std::string basecolor;
    std::string normal;
    std::string height;
    std::string roughness;
    std::string mask;
    std::string heightScale;
    PBRTextureLocation(std::string basecolor = "material.basecolor", std::string normal = "material.normal",
                       std::string height = "heightMap", std::string roughness = "roughness",
                       std::string mask = "mask", std::string heightScale = "heightScale")
        : basecolor(basecolor), normal(normal), height(height), roughness(roughness), mask(mask),
          heightScale(heightScale) {}
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
