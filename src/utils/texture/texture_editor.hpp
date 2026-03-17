#ifndef TEXTURE_EDITOR_HPP
#define TEXTURE_EDITOR_HPP
#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include <IconsFont/IconsLucide.h>

#include "../mesh/model.hpp"
#include "../mesh/solve_uv.hpp"
#include "texture.hpp"

class TextureEditor {
private:
  constexpr static std::string_view textureListPath = PROJECT_DIR "/textures.toml";
  std::string_view _textureListPath;

public:
  explicit TextureEditor(Model &model, bool isPBR = false,
                         const std::string_view textureListPath = TextureEditor::textureListPath,
                         float scaleStep = 0.1f, float scaleMin = 0.1f, float scaleMax = 2.0f);
  ~TextureEditor();

  void renderImage(float repeatSize = 2);
  void handleTextureInput();
  void handleBrushInput(const Camera &camera, float width, float height);
  void controls();

  static constexpr const char *icon = ICON_LC_IMAGE;

private:
  void renderList();

private:
  Model &_model;

private:
  enum class SelectMode : int { Faces, Point };
  SelectMode _selectMode = SelectMode::Point;

private:
  // brush and solving
  int _brushRadius = 10;
  HitResult _hitResult{};

  SolveUV::SolvingMode _solvingMode = SolveUV::SolvingMode::GeodesicSplines;
  bool _solved = false;
  bool _autoCalculate = true;

public:
  inline const SolveUV::SolvingMode solvingMode() const { return _solvingMode; }
  inline const bool isGeodesic() const { return _solvingMode == SolveUV::SolvingMode::GeodesicSplines; }

private:
  // texture
  bool _isPBR = false;
  std::vector<std::unique_ptr<ImageTexture>> _textureList{};
  std::vector<std::unique_ptr<PBRTexture>> _pbrTextureList{};
  // TODO: how to deselect textrue ?
  int _selectedTexture = -1;

public:
  const bool isPBR() const { return _isPBR; }

  bool add(const std::string &imagePath);
  bool add(const std::string &textrueDirectory, float heightScale);
  [[nodiscard]] inline const std::vector<std::unique_ptr<ImageTexture>> &textureList() {
    return _textureList;
  }
  [[nodiscard]] inline const std::vector<std::unique_ptr<PBRTexture>> &pbrTextureList() {
    return _pbrTextureList;
  }

  [[nodiscard]] inline int selected() const { return _selectedTexture; }
  [[nodiscard]] inline const ImageTexture *selectedTexture() const {
    if (_selectedTexture < 0 || _selectedTexture >= _textureList.size())
      return nullptr;
    return _textureList[_selectedTexture].get();
  }
  [[nodiscard]] inline ImageTexture *selectedTexture() {
    if (_selectedTexture < 0 || _selectedTexture >= _textureList.size())
      return nullptr;
    return _textureList[_selectedTexture].get();
  }
  [[nodiscard]] inline PBRTexture *selectedPBR() const {
    if (_selectedTexture < 0 || _selectedTexture >= _pbrTextureList.size())
      return nullptr;
    return _pbrTextureList[_selectedTexture].get();
  }
  inline void setSelected(int i) {
    if (_selectedTexture < _textureList.size() && _selectedTexture >= 0)
      _selectedTexture = i;
  }
  inline void setSelectedPBR(int i) {
    if (_selectedTexture < _pbrTextureList.size() && _selectedTexture >= 0)
      _selectedTexture = i;
  }

private:
  // scale
  float _scale = 1.0f;
  const float _scaleStep = 0.1f;
  const float _scaleMin = 0.1f;
  const float _scaleMax = 2.0f;

  // move
  glm::vec2 _offset{};
  glm::vec2 _offsetAnchor{};
  bool _isMouseLeftDown = false;
  glm::vec2 _mousePosAnchor{};

  // rotate
  float _theta = 0; // in degrees

  enum TextureEditingMode : int { None, Rotate, Scale, Move };
  TextureEditingMode _mode = TextureEditingMode::None;

public:
  [[nodiscard]] inline float scale() const { return _scale; }
  [[nodiscard]] inline glm::vec2 offset() const { return _offset; }
  [[nodiscard]] inline float theta(bool inDegrees = false) const {
    return inDegrees ? _theta : glm::radians(_theta);
  }
};

#endif // !TEXTURE_EDITOR_HPP
