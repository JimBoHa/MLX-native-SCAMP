#pragma once

#include <cstdint>
#include <vector>

#include "scamp/common.h"
#include "scamp/profile.h"

namespace SCAMP {

// Apple-native counterpart of upstream SCAMPArgs. Defaults select the Metal
// path that this initial native library slice implements.
struct SCAMPArgs {
  void validate();
  void print();
  bool InitProfileMemory();

  std::vector<double> timeseries_a;
  std::vector<double> timeseries_b;
  Profile profile_a{PROFILE_TYPE_1NN_INDEX};
  Profile profile_b{PROFILE_TYPE_1NN_INDEX};
  bool has_b{false};
  std::uint64_t window{0};
  std::uint64_t max_tile_size{512000};
  std::int64_t distributed_start_row{-1};
  std::int64_t distributed_start_col{-1};
  double distance_threshold{0.0};
  SCAMPPrecisionType precision_type{PRECISION_SINGLE};
  SCAMPProfileType profile_type{PROFILE_TYPE_1NN_INDEX};
  bool computing_rows{true};
  bool computing_columns{true};
  bool keep_rows_separate{false};
  bool is_aligned{false};
  bool silent_mode{true};
  std::int64_t max_matches_per_column{5};
  std::int64_t matrix_height{50};
  std::int64_t matrix_width{50};
};

} // namespace SCAMP
