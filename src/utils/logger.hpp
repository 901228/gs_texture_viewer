#ifndef LOGGER_HPP
#define LOGGER_HPP
#include "spdlog/common.h"
#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#define LOGGER_CALL(logger, level, ...)                                                                      \
  (logger)->log(spdlog::source_loc{__FILE__, __LINE__, static_cast<const char *>(__FUNCTION__)}, level,      \
                __VA_ARGS__)

#define TRACE(...)                                                                                           \
  SPDLOG_LOGGER_CALL(spdlog::default_logger_raw(), spdlog::level::level_enum::trace, __VA_ARGS__)
#define DEBUG(...)                                                                                           \
  SPDLOG_LOGGER_CALL(spdlog::default_logger_raw(), spdlog::level::level_enum::debug, __VA_ARGS__)
#define INFO(...)                                                                                            \
  SPDLOG_LOGGER_CALL(spdlog::default_logger_raw(), spdlog::level::level_enum::info, __VA_ARGS__)
#define WARN(...)                                                                                            \
  SPDLOG_LOGGER_CALL(spdlog::default_logger_raw(), spdlog::level::level_enum::warn, __VA_ARGS__)
#define ERROR(...)                                                                                           \
  SPDLOG_LOGGER_CALL(spdlog::default_logger_raw(), spdlog::level::level_enum::err, __VA_ARGS__)
#define CRTICAL(...)                                                                                         \
  SPDLOG_LOGGER_CALL(spdlog::default_logger_raw(), spdlog::level::level_enum::critical, __VA_ARGS__)

namespace Utils {

inline void InitLogger(spdlog::level::level_enum level = spdlog::level::debug) {

  const char *pattern = "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%s:%#] %v";

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(level);
  console_sink->set_pattern(pattern);

  auto file_sink = std::make_shared<spdlog::sinks::daily_file_format_sink_mt>("logs/daily.txt", 2, 30);
  file_sink->set_level(spdlog::level::trace);
  console_sink->set_pattern(pattern);

  auto logger =
      std::make_shared<spdlog::logger>("GS_Viewer", spdlog::sinks_init_list{console_sink, file_sink});
  logger->set_level(level);
  spdlog::set_default_logger(logger);
}

} // namespace Utils

#endif // !LOGGER_HPP
