#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

#include "scamp/common.h"

namespace SCAMP {

struct ProfileData {
  std::vector<std::uint32_t> uint32_value;
  std::vector<std::uint64_t> uint64_value;
  std::vector<float> float_value;
  std::vector<double> double_value;
  std::vector<std::vector<float>> matrix_value;
  std::vector<
      std::priority_queue<SCAMPmatch, std::vector<SCAMPmatch>, compareMatch>>
      match_value;
  std::vector<SCAMPmatch> match_value_unordered;
};

class Profile {
public:
  Profile() = default;
  explicit Profile(SCAMPProfileType profile_type) : type(profile_type) {}
  Profile(SCAMPProfileType profile_type, std::size_t size,
          float threshold = 0.0F, std::int64_t matrix_width = -1,
          std::int64_t matrix_height = -1);

  void Alloc(std::size_t size, std::int64_t matrix_height,
             std::int64_t matrix_width, float default_threshold);

  std::vector<ProfileData> data;
  std::vector<float> thresholds;
  SCAMPProfileType type{PROFILE_TYPE_INVALID};
};

} // namespace SCAMP
