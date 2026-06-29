#include "main_window.hpp"
#include "utils/logger.hpp"

int main(int argc, char **argv) {

  Utils::InitLogger();

  MainWindow mainWindow{};

  return 0;
}
