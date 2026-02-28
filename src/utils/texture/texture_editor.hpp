#ifndef TEXTURE_EDITOR_HPP
#define TEXTURE_EDITOR_HPP
#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "../texture/texture.hpp"

class TextureEditor {
private:
  constexpr static std::string_view textureListPath = "textures.toml";

public:
  explicit TextureEditor(const std::string &textureListPath = "textures.toml", float scaleStep = 0.1f,
                         float scaleMin = 0.1f, float scaleMax = 2.0f);
  ~TextureEditor();

  void renderImage(float repeatSize = 2);
  void renderList();
  void handleInput();

private:
  // texture
  std::vector<std::unique_ptr<ImageTexture>> _textureList{};
  int _selectedTexture = -1;

public:
  bool add(const std::string &imagePath);
  [[nodiscard]] inline const std::vector<std::unique_ptr<ImageTexture>> &textureList() {
    return _textureList;
  }

  [[nodiscard]] inline int selected() const { return _selectedTexture; }
  [[nodiscard]] inline ImageTexture *selectedTexture() {
    if (_selectedTexture < 0 || _selectedTexture >= _textureList.size())
      return nullptr;
    return _textureList[_selectedTexture].get();
  }
  inline void setSelected(int i) {
    if (_selectedTexture < _textureList.size() && _selectedTexture >= 0)
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
  float _theta = 0;

  enum TextureEditingMode : int { None, Rotate, Scale, Move };
  TextureEditingMode _mode = TextureEditingMode::None;

public:
  [[nodiscard]] inline float scale() const { return _scale; }
  [[nodiscard]] inline glm::vec2 offset() const { return _offset; }
  [[nodiscard]] inline float theta() const { return _theta; }
};

#endif // !TEXTURE_EDITOR_HPP
