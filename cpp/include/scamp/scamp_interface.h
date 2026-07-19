#pragma once

#include <vector>

#include "scamp/scamp_args.h"

namespace SCAMP {

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices,
              int num_threads);
void do_SCAMP(SCAMPArgs *args);

// Retained for source compatibility. On this port it reports the number of
// MLX Metal devices, not CUDA devices.
int num_available_gpus();

} // namespace SCAMP
