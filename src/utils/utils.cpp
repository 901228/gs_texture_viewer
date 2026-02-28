#include "utils.hpp"

#include <nfd.h>

#include "logger.hpp"

std::string Utils::FileDialog::openImageDialog() {

  std::string filepath;

  nfdu8filteritem_t filters[1] = {{"Image", "jpg,JPG,jpeg,JPEG,png,PNG"}};
  nfdopendialogu8args_t args = {nullptr};
  args.filterList = filters;
  args.filterCount = 1;

  nfdu8char_t *outPath;
  nfdresult_t result = NFD_OpenDialogU8_With(&outPath, &args);
  if (result == NFD_OKAY) {
    filepath = {outPath};
    NFD_FreePathU8(outPath);
  } else if (result == NFD_CANCEL) {
    // cancel
  } else {
    ERROR("Error: {}", NFD_GetError());
  }

  return filepath;
}
