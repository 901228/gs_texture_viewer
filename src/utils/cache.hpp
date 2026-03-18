#ifndef UTILS_CACHE_HPP
#define UTILS_CACHE_HPP
#pragma once

#include <list>
#include <optional>
#include <shared_mutex>
#include <unordered_map>

namespace Cache {

template <typename T> void hash_combine(size_t &seed, const T &val) {
  seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename K, typename V, typename Hash = std::hash<K>, typename KeyEqual = std::equal_to<K>>
class LRUCache {
  size_t capacity_;
  std::list<std::pair<K, V>> order_;
  std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator, Hash, KeyEqual> map_;
  mutable std::shared_mutex mtx_;

public:
  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  std::optional<V> get(const K &key) {
    // `get` has to modify `order_`, so it still uses `unique_lock`
    std::unique_lock<std::shared_mutex> lock(mtx_);
    auto it = map_.find(key);
    if (it == map_.end())
      return std::nullopt;

    // move to front (most recently used)
    order_.splice(order_.begin(), order_, it->second);
    return it->second->second;
  }

  // read-only query (does not update LRU order, suitable for peek usage)
  std::optional<V> peek(const K &key) const {
    std::shared_lock<std::shared_mutex> lock(mtx_); // multiple peeks can be executed in parallel
    auto it = map_.find(key);
    if (it == map_.end())
      return std::nullopt;
    return it->second->second;
  }

  void put(const K &key, const V &val) {
    std::unique_lock<std::shared_mutex> lock(mtx_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      it->second->second = val;
      order_.splice(order_.begin(), order_, it->second);
      return;
    }
    if (map_.size() >= capacity_) {
      map_.erase(order_.back().first);
      order_.pop_back();
    }
    order_.emplace_front(key, val);
    map_[key] = order_.begin();
  }

  bool contains(const K &key) const {
    std::shared_lock<std::shared_mutex> lock(mtx_);
    return map_.count(key) > 0;
  }

  size_t size() const {
    std::shared_lock<std::shared_mutex> lock(mtx_);
    return map_.size();
  }
};

} // namespace Cache

#endif // !UTILS_CACHE_HPP
