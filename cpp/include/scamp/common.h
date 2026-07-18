#pragma once

#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace SCAMP {

enum SCAMPProfileType {
  PROFILE_TYPE_INVALID = 0,
  PROFILE_TYPE_1NN_INDEX = 1,
  PROFILE_TYPE_SUM_THRESH = 2,
  PROFILE_TYPE_FREQUENCY_THRESH = 3,
  PROFILE_TYPE_KNN = 4,
  PROFILE_TYPE_1NN_MULTIDIM = 5,
  PROFILE_TYPE_1NN = 6,
  PROFILE_TYPE_APPROX_ALL_NEIGHBORS = 7,
  PROFILE_TYPE_MATRIX_SUMMARY = 8,
};

enum SCAMPPrecisionType {
  PRECISION_INVALID = 0,
  PRECISION_SINGLE = 1,
  PRECISION_MIXED = 2,
  PRECISION_DOUBLE = 3,
  PRECISION_ULTRA = 4,
};

union mp_entry {
  float floats[2];
  std::uint32_t ints[2];
  std::uint64_t ulong;
};

struct SCAMPmatch {
  SCAMPmatch() = default;
  SCAMPmatch(float correlation, std::uint32_t row_index,
             std::uint32_t column_index)
      : corr(correlation), row(row_index), col(column_index) {}

  bool operator<(const SCAMPmatch &other) const {
    return col == other.col ? corr > other.corr : col < other.col;
  }

  float corr{-2.0F};
  std::uint32_t row{0};
  std::uint32_t col{0};
};

class compareMatch {
public:
  bool operator()(const SCAMPmatch &lhs, const SCAMPmatch &rhs) const {
    return lhs.corr > rhs.corr;
  }
};

class SCAMPException : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

std::string GetPrecisionTypeString(SCAMPPrecisionType type);
std::string GetProfileTypeString(SCAMPProfileType type);

float GetProfileCorrelation(std::uint64_t packed_profile);
std::int32_t GetProfileIndex(std::uint64_t packed_profile);

} // namespace SCAMP
