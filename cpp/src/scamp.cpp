#include "scamp/scamp.h"

#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace SCAMP {
namespace {

namespace mx = mlx::core;

constexpr double kFlatnessEpsilon = 1e-13;
constexpr float kInvalidCorrelation = -2.0F;
constexpr std::uint32_t kIndexInitializer = 0x80000000U;

struct PreparedSeries {
  std::vector<double> clean;
  std::vector<double> means;
  std::vector<float> inv_norm;
  std::vector<float> df;
  std::vector<float> dg;
};

const char *kProfileSource = R"METAL(
    uint diag_slot = thread_position_in_grid.x;
    uint n_a = inv_norm_a_shape[0];
    uint n_b = inv_norm_b_shape[0];
    uint exclusion = uint(config[0]);
    bool self_join = config[1] != 0;
    bool compute_cols = config[2] != 0;
    bool compute_rows = config[3] != 0;
    bool aligned = config[4] != 0;
    int start_col = config[5];
    int start_row = config[6];

    int diagonal;
    if (self_join) {
        diagonal = int(exclusion + diag_slot);
    } else {
        diagonal = int(diag_slot) - int(n_a - 1);
    }

    uint col = diagonal < 0 ? uint(-diagonal) : 0;
    uint row = diagonal > 0 ? uint(diagonal) : 0;
    if (self_join) {
        col = uint(diagonal);
        row = 0;
    }
    uint diagonal_length = metal::min(n_a - col, n_b - row);

    float covariance = initial_cov[diag_slot];

    for (uint step = 0; step < diagonal_length; ++step, ++col, ++row) {
        bool excluded = aligned &&
            metal::abs((int(col) + start_col) - (int(row) + start_row)) <
                int(exclusion);
        float norm_product = inv_norm_a[col] * inv_norm_b[row];
        if (!excluded && norm_product > 0.0f) {
            float corr = covariance * norm_product;
            if (metal::isfinite(corr)) {
                corr = metal::fmin(1.0f, metal::fmax(-1.0f, corr));
                uint bits = as_type<uint>(corr);
                uint key = (bits & 0x80000000u) ? ~bits :
                                                     (bits ^ 0x80000000u);
                if (compute_cols) {
                    atomic_fetch_max_explicit(
                        &best_a[col], key, memory_order_relaxed);
                }
                if (self_join) {
                    atomic_fetch_max_explicit(
                        &best_a[row], key, memory_order_relaxed);
                } else if (compute_rows) {
                    atomic_fetch_max_explicit(
                        &best_b[row], key, memory_order_relaxed);
                }
            }
        }
        if (step + 1 < diagonal_length) {
            covariance += df_a[col] * dg_b[row] +
                          dg_a[col] * df_b[row];
        }
    }
)METAL";

const char *kIndexSource = R"METAL(
    uint diag_slot = thread_position_in_grid.x;
    uint n_a = inv_norm_a_shape[0];
    uint n_b = inv_norm_b_shape[0];
    uint exclusion = uint(config[0]);
    bool self_join = config[1] != 0;
    bool compute_cols = config[2] != 0;
    bool compute_rows = config[3] != 0;
    bool aligned = config[4] != 0;
    int start_col = config[5];
    int start_row = config[6];

    int diagonal;
    if (self_join) {
        diagonal = int(exclusion + diag_slot);
    } else {
        diagonal = int(diag_slot) - int(n_a - 1);
    }

    uint col = diagonal < 0 ? uint(-diagonal) : 0;
    uint row = diagonal > 0 ? uint(diagonal) : 0;
    if (self_join) {
        col = uint(diagonal);
        row = 0;
    }
    uint diagonal_length = metal::min(n_a - col, n_b - row);

    float covariance = initial_cov[diag_slot];

    for (uint step = 0; step < diagonal_length; ++step, ++col, ++row) {
        bool excluded = aligned &&
            metal::abs((int(col) + start_col) - (int(row) + start_row)) <
                int(exclusion);
        float norm_product = inv_norm_a[col] * inv_norm_b[row];
        if (!excluded && norm_product > 0.0f) {
            float corr = covariance * norm_product;
            if (metal::isfinite(corr)) {
                corr = metal::fmin(1.0f, metal::fmax(-1.0f, corr));
                uint bits = as_type<uint>(corr);
                uint key = (bits & 0x80000000u) ? ~bits :
                                                     (bits ^ 0x80000000u);
                if (compute_cols && best_a[col] == key) {
                    atomic_fetch_min_explicit(
                        &index_a[col], row, memory_order_relaxed);
                }
                if (self_join && best_a[row] == key) {
                    atomic_fetch_min_explicit(
                        &index_a[row], col, memory_order_relaxed);
                } else if (!self_join && compute_rows && best_b[row] == key) {
                    atomic_fetch_min_explicit(
                        &index_b[row], col, memory_order_relaxed);
                }
            }
        }
        if (step + 1 < diagonal_length) {
            covariance += df_a[col] * dg_b[row] +
                          dg_a[col] * df_b[row];
        }
    }
)METAL";

auto &ProfileKernel() {
  static auto kernel = mx::fast::metal_kernel(
      "mlx_scamp_cpp_1nn_profile",
      {"initial_cov", "inv_norm_a", "inv_norm_b", "df_a", "df_b", "dg_a",
       "dg_b", "config"},
      {"best_a", "best_b"}, kProfileSource, "", true, true);
  return kernel;
}

auto &IndexKernel() {
  static auto kernel = mx::fast::metal_kernel(
      "mlx_scamp_cpp_1nn_index",
      {"initial_cov", "inv_norm_a", "inv_norm_b", "df_a", "df_b", "dg_a",
       "dg_b", "config", "best_a", "best_b"},
      {"index_a", "index_b"}, kIndexSource, "", true, true);
  return kernel;
}

std::vector<double> MovingMean(const std::vector<double> &values,
                               std::size_t window) {
  const std::size_t output_size = values.size() - window + 1;
  std::vector<double> result(output_size, 0.0);
  double primary = values.front();
  double compensation = 0.0;
  for (std::size_t i = 1; i < window; ++i) {
    const double sum = primary + values[i];
    const double recovered = sum - primary;
    compensation += (primary - (sum - recovered)) + (values[i] - recovered);
    primary = sum;
  }
  result[0] = (primary + compensation) / static_cast<double>(window);

  for (std::size_t i = window; i < values.size(); ++i) {
    double sum = primary - values[i - window];
    double recovered = sum - primary;
    compensation +=
        (primary - (sum - recovered)) - (values[i - window] + recovered);
    primary = sum;

    sum = primary + values[i];
    recovered = sum - primary;
    compensation += (primary - (sum - recovered)) + (values[i] - recovered);
    primary = sum;
    result[i - window + 1] =
        (primary + compensation) / static_cast<double>(window);
  }
  return result;
}

PreparedSeries PrepareSeries(const std::vector<double> &input,
                             std::size_t window) {
  std::vector<double> clean_double(input.size(), 0.0);
  std::vector<int> invalid_prefix(input.size() + 1, 0);
  double origin = 0.0;
  const auto first_finite =
      std::find_if(input.begin(), input.end(), [](double value) {
        return std::isfinite(value);
      });
  if (first_finite != input.end()) {
    origin = *first_finite;
  }
  for (std::size_t i = 0; i < input.size(); ++i) {
    if (std::isfinite(input[i])) {
      // Pearson correlation is independently translation-invariant for each
      // series.  Removing a stable origin before the rolling statistics keeps
      // nearby variation representable when the raw baseline is very large.
      clean_double[i] = input[i] - origin;
      if (!std::isfinite(clean_double[i])) {
        throw SCAMPException(
            "Error: input range overflows stable centering precomputation");
      }
    }
    invalid_prefix[i + 1] =
        invalid_prefix[i] + (std::isfinite(input[i]) ? 0 : 1);
  }

  const std::size_t subsequences = input.size() - window + 1;
  std::vector<double> means_double = MovingMean(clean_double, window);
  std::vector<double> norm_squared(subsequences, 0.0);
  for (std::size_t j = 0; j < window; ++j) {
    const double centered = clean_double[j] - means_double[0];
    norm_squared[0] += centered * centered;
  }
  for (std::size_t i = 1; i < subsequences; ++i) {
    norm_squared[i] = norm_squared[i - 1] +
                      ((clean_double[i - 1] - means_double[i - 1]) +
                       (clean_double[i + window - 1] - means_double[i])) *
                          (clean_double[i + window - 1] - clean_double[i - 1]);
  }

  PreparedSeries result;
  result.clean = std::move(clean_double);
  result.means = std::move(means_double);
  result.inv_norm.resize(subsequences, 0.0F);
  result.df.resize(subsequences, 0.0F);
  result.dg.resize(subsequences, 0.0F);

  for (std::size_t i = 0; i < subsequences; ++i) {
    if (!std::isfinite(result.means[i])) {
      throw SCAMPException(
          "Error: input magnitude overflows the moving-mean precomputation");
    }
    const bool finite_window = invalid_prefix[i + window] == invalid_prefix[i];
    if (finite_window && norm_squared[i] > kFlatnessEpsilon &&
        std::isfinite(norm_squared[i])) {
      result.inv_norm[i] = static_cast<float>(1.0 / std::sqrt(norm_squared[i]));
    }
  }
  for (std::size_t i = 0; i + 1 < subsequences; ++i) {
    const double df = (result.clean[i + window] - result.clean[i]) * 0.5;
    const double dg = (result.clean[i + window] - result.means[i + 1]) +
                      (result.clean[i] - result.means[i]);
    if (!std::isfinite(df) || !std::isfinite(dg) ||
        std::abs(df) > std::numeric_limits<float>::max() ||
        std::abs(dg) > std::numeric_limits<float>::max()) {
      throw SCAMPException(
          "Error: input magnitude overflows the float32 Metal recurrence");
    }
    result.df[i] = static_cast<float>(df);
    result.dg[i] = static_cast<float>(dg);
  }
  return result;
}

std::vector<float> InitialCovariances(const PreparedSeries &a,
                                      const PreparedSeries &b,
                                      std::size_t window, bool self_join,
                                      std::size_t exclusion) {
  const std::size_t n_a = a.means.size();
  const std::size_t n_b = b.means.size();
  const std::size_t count =
      self_join ? (n_a > exclusion ? n_a - exclusion : 0) : n_a + n_b - 1;
  std::vector<float> result(count, 0.0F);
  for (std::size_t slot = 0; slot < count; ++slot) {
    std::int64_t diagonal = 0;
    if (self_join) {
      diagonal = static_cast<std::int64_t>(exclusion + slot);
    } else {
      diagonal =
          static_cast<std::int64_t>(slot) - static_cast<std::int64_t>(n_a - 1);
    }
    std::size_t col = diagonal < 0 ? static_cast<std::size_t>(-diagonal) : 0;
    std::size_t row = diagonal > 0 ? static_cast<std::size_t>(diagonal) : 0;
    if (self_join) {
      col = static_cast<std::size_t>(diagonal);
      row = 0;
    }
    double covariance = 0.0;
    for (std::size_t k = 0; k < window; ++k) {
      covariance +=
          (a.clean[col + k] - a.means[col]) * (b.clean[row + k] - b.means[row]);
    }
    if (!std::isfinite(covariance) ||
        std::abs(covariance) > std::numeric_limits<float>::max()) {
      throw SCAMPException(
          "Error: initial covariance exceeds the float32 Metal range");
    }
    result[slot] = static_cast<float>(covariance);
  }
  return result;
}

float MaxMagnitude(const std::vector<float> &values) {
  float maximum = 0.0F;
  for (float value : values) {
    maximum = std::max(maximum, std::abs(value));
  }
  return maximum;
}

void ValidateRecurrenceRange(const PreparedSeries &a, const PreparedSeries &b,
                             const std::vector<float> &initial_covariances) {
  const long double max_step =
      static_cast<long double>(MaxMagnitude(a.df)) * MaxMagnitude(b.dg) +
      static_cast<long double>(MaxMagnitude(a.dg)) * MaxMagnitude(b.df);
  const long double bound =
      static_cast<long double>(MaxMagnitude(initial_covariances)) +
      static_cast<long double>(std::max(a.means.size(), b.means.size())) *
          max_step;
  if (!std::isfinite(bound) || bound > std::numeric_limits<float>::max()) {
    throw SCAMPException(
        "Error: input magnitude can overflow the float32 Metal recurrence");
  }
}

mx::array ToArray(const std::vector<float> &values) {
  return mx::array(values.begin(),
                   mx::Shape{static_cast<mx::ShapeElem>(values.size())},
                   mx::float32);
}

float DecodeOrderedKey(std::uint32_t key) {
  const std::uint32_t bits =
      (key & 0x80000000U) != 0U ? key ^ 0x80000000U : ~key;
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint64_t PackProfile(float correlation, std::uint32_t index) {
  mp_entry entry{};
  entry.floats[0] = correlation;
  entry.ints[1] = index;
  return entry.ulong;
}

void MaterializeProfile(const mx::array &keys, const mx::array &indices,
                        std::vector<std::uint64_t> *output) {
  auto evaluated_keys = keys;
  auto evaluated_indices = indices;
  mx::eval(evaluated_keys, evaluated_indices);
  const auto *key_data = evaluated_keys.data<std::uint32_t>();
  const auto *index_data = evaluated_indices.data<std::uint32_t>();
  output->resize(evaluated_keys.size());
  for (std::size_t i = 0; i < output->size(); ++i) {
    if (index_data[i] == kIndexInitializer) {
      (*output)[i] = PackProfile(kInvalidCorrelation, 0xFFFFFFFFU);
    } else {
      (*output)[i] = PackProfile(DecodeOrderedKey(key_data[i]), index_data[i]);
    }
  }
}

void ValidateResources(const std::vector<int> &devices, int num_threads) {
  if (num_threads < 0) {
    throw SCAMPException("Error: num_threads must not be negative");
  }
  if (num_threads != 0) {
    throw SCAMPException(
        "Error: native C++ CPU workers are not implemented yet; request "
        "Metal device 0 with num_threads=0");
  }
  if (devices.size() != 1 || devices.front() != 0) {
    throw SCAMPException(
        "Error: MLX on Apple Silicon exposes one Metal GPU; devices must be "
        "{0}");
  }
  if (!mx::metal::is_available()) {
    throw SCAMPException("Error: MLX Metal device 0 is unavailable");
  }
}

} // namespace

std::string GetPrecisionTypeString(SCAMPPrecisionType type) {
  switch (type) {
  case PRECISION_SINGLE:
    return "PRECISION_SINGLE";
  case PRECISION_MIXED:
    return "PRECISION_MIXED";
  case PRECISION_DOUBLE:
    return "PRECISION_DOUBLE";
  case PRECISION_ULTRA:
    return "PRECISION_ULTRA";
  case PRECISION_INVALID:
    return "PRECISION_INVALID";
  }
  return "PRECISION_UNKNOWN";
}

std::string GetProfileTypeString(SCAMPProfileType type) {
  switch (type) {
  case PROFILE_TYPE_1NN_INDEX:
    return "PROFILE_TYPE_1NN_INDEX";
  case PROFILE_TYPE_SUM_THRESH:
    return "PROFILE_TYPE_SUM_THRESH";
  case PROFILE_TYPE_FREQUENCY_THRESH:
    return "PROFILE_TYPE_FREQUENCY_THRESH";
  case PROFILE_TYPE_KNN:
    return "PROFILE_TYPE_KNN";
  case PROFILE_TYPE_1NN_MULTIDIM:
    return "PROFILE_TYPE_1NN_MULTIDIM";
  case PROFILE_TYPE_1NN:
    return "PROFILE_TYPE_1NN";
  case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
    return "PROFILE_TYPE_APPROX_ALL_NEIGHBORS";
  case PROFILE_TYPE_MATRIX_SUMMARY:
    return "PROFILE_TYPE_MATRIX_SUMMARY";
  case PROFILE_TYPE_INVALID:
    return "PROFILE_TYPE_INVALID";
  }
  return "PROFILE_TYPE_UNKNOWN";
}

float GetProfileCorrelation(std::uint64_t packed_profile) {
  mp_entry entry{};
  entry.ulong = packed_profile;
  return entry.floats[0];
}

std::int32_t GetProfileIndex(std::uint64_t packed_profile) {
  mp_entry entry{};
  entry.ulong = packed_profile;
  std::int32_t index = -1;
  static_assert(sizeof(index) == sizeof(entry.ints[1]));
  std::memcpy(&index, &entry.ints[1], sizeof(index));
  return index;
}

Profile::Profile(SCAMPProfileType profile_type, std::size_t size,
                 float threshold, std::int64_t matrix_width,
                 std::int64_t matrix_height)
    : type(profile_type) {
  Alloc(size, matrix_height, matrix_width, threshold);
}

void Profile::Alloc(std::size_t size, std::int64_t matrix_height,
                    std::int64_t matrix_width, float default_threshold) {
  data.clear();
  thresholds.clear();
  data.emplace_back();
  switch (type) {
  case PROFILE_TYPE_1NN_INDEX:
    data[0].uint64_value.assign(size,
                                PackProfile(kInvalidCorrelation, 0xFFFFFFFFU));
    break;
  case PROFILE_TYPE_SUM_THRESH:
    data[0].double_value.assign(size, 0.0);
    break;
  case PROFILE_TYPE_1NN:
    data[0].float_value.assign(size, kInvalidCorrelation);
    break;
  case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
    data[0].match_value.resize(size);
    thresholds.assign(size, default_threshold);
    break;
  case PROFILE_TYPE_MATRIX_SUMMARY:
    if (matrix_height <= 0 || matrix_width <= 0) {
      throw SCAMPException("Error: matrix profile dimensions must be positive");
    }
    data[0].float_value.assign(
        static_cast<std::size_t>(matrix_height * matrix_width),
        kInvalidCorrelation);
    break;
  default:
    data.clear();
    break;
  }
}

void SCAMPArgs::validate() {
  if (profile_type != PROFILE_TYPE_1NN_INDEX) {
    throw SCAMPException(
        "Error: native C++ support currently covers PROFILE_TYPE_1NN_INDEX "
        "only");
  }
  if (precision_type != PRECISION_SINGLE) {
    throw SCAMPException(
        "Error: the native Metal kernel currently supports PRECISION_SINGLE "
        "only");
  }
  if (window < 3) {
    throw SCAMPException("Error: Subsequence length must be at least 3");
  }
  if (max_tile_size < 1024) {
    throw SCAMPException("Error: max tile size must be at least 1024");
  }
  if (max_tile_size / 2 < window) {
    throw SCAMPException(
        "Error: Tile length and width must be at least 2x larger than the "
        "window size");
  }
  if (timeseries_a.size() < window || (has_b && timeseries_b.size() < window)) {
    throw SCAMPException(
        "Error: Input time series must be at least the window size");
  }
  if (!computing_columns && !computing_rows) {
    throw SCAMPException(
        "Error: at least one of computing_columns or computing_rows must be "
        "enabled");
  }
  if (!has_b && (!computing_columns || !computing_rows || keep_rows_separate)) {
    throw SCAMPException(
        "Error: self joins require both directions and a merged profile");
  }
  if (has_b && computing_rows != keep_rows_separate) {
    throw SCAMPException(
        "Error: AB row profiles require computing_rows=true and "
        "keep_rows_separate=true together");
  }
  if (distributed_start_row < -1 || distributed_start_col < -1) {
    throw SCAMPException(
        "Error: distributed offsets must be -1 (unset) or nonnegative");
  }
  const bool row_offset_set = distributed_start_row != -1;
  const bool col_offset_set = distributed_start_col != -1;
  if (row_offset_set != col_offset_set) {
    throw SCAMPException(
        "Error: distributed row and column offsets must be set together");
  }
  if ((row_offset_set &&
       distributed_start_row > std::numeric_limits<std::int32_t>::max()) ||
      (col_offset_set &&
       distributed_start_col > std::numeric_limits<std::int32_t>::max())) {
    throw SCAMPException(
        "Error: distributed offsets exceed the Metal int32 index range");
  }
  if (row_offset_set && !has_b) {
    throw SCAMPException(
        "Error: distributed offsets are supported only for aligned AB joins");
  }
  if (row_offset_set && has_b && !is_aligned) {
    throw SCAMPException(
        "Error: 1NN distributed offsets require is_aligned=true");
  }
  if (row_offset_set) {
    const auto max_metal_index =
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    const auto columns = timeseries_a.size() - window + 1;
    const auto rows = timeseries_b.size() - window + 1;
    if (static_cast<std::uint64_t>(distributed_start_col) + columns - 1 >
            max_metal_index ||
        static_cast<std::uint64_t>(distributed_start_row) + rows - 1 >
            max_metal_index) {
      throw SCAMPException(
          "Error: distributed tile bounds exceed the Metal int32 index range");
    }
  }
  const auto max_shape =
      static_cast<std::size_t>(std::numeric_limits<mx::ShapeElem>::max());
  if (timeseries_a.size() > max_shape ||
      (has_b && timeseries_b.size() > max_shape)) {
    throw SCAMPException("Error: input exceeds MLX's int32 shape limit");
  }
}

void SCAMPArgs::print() {
  std::cout << "window: " << window << '\n'
            << "max_tile_size: " << max_tile_size << '\n'
            << "has_b: " << has_b << '\n'
            << "keep_rows_separate: " << keep_rows_separate << '\n'
            << "distributed_start_row: " << distributed_start_row << '\n'
            << "distributed_start_col: " << distributed_start_col << '\n'
            << "computing_rows: " << computing_rows << '\n'
            << "computing_columns: " << computing_columns << '\n'
            << "is_aligned: " << is_aligned << '\n'
            << "profile_type: " << GetProfileTypeString(profile_type) << '\n'
            << "precision_type: " << GetPrecisionTypeString(precision_type)
            << '\n'
            << "distance_threshold: " << distance_threshold << '\n'
            << "silent_mode: " << silent_mode << '\n'
            << "max_matches_per_column: " << max_matches_per_column << '\n'
            << "timeseries_a size: " << timeseries_a.size() << '\n'
            << "timeseries_b size: " << timeseries_b.size() << std::endl;
}

bool SCAMPArgs::InitProfileMemory() {
  if (timeseries_a.size() < window || (has_b && timeseries_b.size() < window)) {
    return false;
  }
  profile_a.type = profile_type;
  profile_b.type = profile_type;
  profile_a.Alloc(timeseries_a.size() - window + 1, matrix_height, matrix_width,
                  static_cast<float>(distance_threshold));
  if (has_b && computing_rows && keep_rows_separate) {
    profile_b.Alloc(timeseries_b.size() - window + 1, matrix_height,
                    matrix_width, static_cast<float>(distance_threshold));
  } else {
    profile_b.data.clear();
    profile_b.thresholds.clear();
  }
  return true;
}

int num_available_gpus() { return mx::metal::is_available() ? 1 : 0; }

void do_SCAMP(SCAMPArgs *args) {
  if (num_available_gpus() <= 0) {
    throw SCAMPException("Error: no MLX Metal device is available");
  }
  do_SCAMP(args, {0}, 0);
}

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices,
              int num_threads) {
  if (args == nullptr) {
    throw SCAMPException("Error: Invalid arguments provided to SCAMP");
  }
  ValidateResources(devices, num_threads);
  if (!args->silent_mode) {
    std::cout << "Validating SCAMP args.\n";
  }
  args->validate();
  if (!args->InitProfileMemory()) {
    throw SCAMPException("Error: Invalid arguments provided to SCAMP");
  }

  const auto start = std::chrono::steady_clock::now();
  PreparedSeries prepared_a = PrepareSeries(args->timeseries_a, args->window);
  std::optional<PreparedSeries> prepared_b_storage;
  if (args->has_b) {
    prepared_b_storage.emplace(PrepareSeries(args->timeseries_b, args->window));
  }
  const PreparedSeries &prepared_b =
      prepared_b_storage.has_value() ? *prepared_b_storage : prepared_a;

  const std::size_t n_a = prepared_a.means.size();
  const std::size_t n_b = prepared_b.means.size();
  const std::uint32_t exclusion =
      static_cast<std::uint32_t>((args->window + 3) / 4);
  const bool self_join = !args->has_b;
  const std::size_t diagonal_count =
      self_join ? (n_a > exclusion ? n_a - exclusion : 0) : n_a + n_b - 1;
  if (diagonal_count == 0) {
    return;
  }
  if (diagonal_count >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw SCAMPException("Error: diagonal count exceeds Metal grid limits");
  }
  auto initial_covariances = InitialCovariances(
      prepared_a, prepared_b, args->window, self_join, exclusion);
  ValidateRecurrenceRange(prepared_a, prepared_b, initial_covariances);

  std::vector<std::int32_t> config = {
      static_cast<std::int32_t>(exclusion),
      self_join ? 1 : 0,
      args->computing_columns ? 1 : 0,
      args->computing_rows ? 1 : 0,
      (self_join || args->is_aligned) ? 1 : 0,
      args->distributed_start_col >= 0
          ? static_cast<std::int32_t>(args->distributed_start_col)
          : 0,
      args->distributed_start_row >= 0
          ? static_cast<std::int32_t>(args->distributed_start_row)
          : 0,
  };

  auto inv_norm_a = ToArray(prepared_a.inv_norm);
  auto df_a = ToArray(prepared_a.df);
  auto dg_a = ToArray(prepared_a.dg);
  auto inv_norm_b = self_join ? inv_norm_a : ToArray(prepared_b.inv_norm);
  auto df_b = self_join ? df_a : ToArray(prepared_b.df);
  auto dg_b = self_join ? dg_a : ToArray(prepared_b.dg);

  std::vector<mx::array> inputs = {
      ToArray(initial_covariances),
      inv_norm_a,
      inv_norm_b,
      df_a,
      df_b,
      dg_a,
      dg_b,
      mx::array(config.begin(),
                mx::Shape{static_cast<mx::ShapeElem>(config.size())},
                mx::int32),
  };
  const int grid = static_cast<int>(diagonal_count);
  const int threadgroup = std::min(256, grid);
  const mx::Device gpu(mx::Device::gpu, 0);
  const auto output_a_size =
      args->computing_columns ? static_cast<mx::ShapeElem>(n_a) : 1;
  const auto output_b_size =
      args->has_b && args->computing_rows ? static_cast<mx::ShapeElem>(n_b) : 1;
  auto best = ProfileKernel()(
      inputs, {mx::Shape{output_a_size}, mx::Shape{output_b_size}},
      {mx::uint32, mx::uint32}, {grid, 1, 1}, {threadgroup, 1, 1}, {}, 0.0F,
      false, gpu);

  inputs.push_back(best[0]);
  inputs.push_back(best[1]);
  auto indices = IndexKernel()(
      inputs, {mx::Shape{output_a_size}, mx::Shape{output_b_size}},
      {mx::uint32, mx::uint32}, {grid, 1, 1}, {threadgroup, 1, 1}, {},
      static_cast<float>(kIndexInitializer), false, gpu);

  if (args->computing_columns) {
    MaterializeProfile(best[0], indices[0],
                       &args->profile_a.data[0].uint64_value);
  }
  if (args->has_b && args->computing_rows) {
    MaterializeProfile(best[1], indices[1],
                       &args->profile_b.data[0].uint64_value);
  }

  if (!args->silent_mode) {
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Finished MLX Metal SCAMP join in " << std::fixed
              << std::setprecision(6) << seconds << " seconds\n";
  }
}

} // namespace SCAMP
