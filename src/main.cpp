// #if defined(_MSC_VER) && (_MSC_VER >= 1900) &&
// !defined(IMGUI_DISABLE_WIN32_FUNCTIONS) #pragma comment(lib,
// "legacy_stdio_definitions") #endif

#include <filesystem>

#include <CLI/CLI.hpp>

#include "main_window.hpp"
#include "utils/logger.hpp"

int parseArgs(int argc, char **argv, std::string &gaussianFile) {
  CLI::App app{"Gaussian Viewer"};

  // add options
  app.add_option("file", gaussianFile, "3DGS PLY file")->required()->check(CLI::ExistingFile);

  // parse
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  // post process
  gaussianFile = std::filesystem::absolute(gaussianFile).string();

  return 0;
}

int main(int argc, char **argv) {

  Utils::InitLogger();

  // std::string gaussianFile;
  // int ret = parseArgs(argc, argv, gaussianFile);
  // if (ret != 0)
  //   return ret;

  // printf("Gaussian file: %s\n", gaussianFile.c_str());

  MainWindow mainWindow{};

  return 0;
}
