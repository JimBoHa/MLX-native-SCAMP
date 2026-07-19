#include "native_cli.h"

#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  try {
    return mlx_scamp::cli::Run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "mlx-scamp-native: error: " << error.what() << '\n';
    return 1;
  }
}
