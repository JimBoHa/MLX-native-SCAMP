#include <scamp/scamp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ExpectedProfile {
  std::vector<float> correlation;
  std::vector<std::int32_t> index;
};

ExpectedProfile BruteForce(const std::vector<double> &a,
                           const std::vector<double> &b, std::size_t window,
                           bool self_join, bool columns, bool aligned = false,
                           std::int64_t start_col = 0,
                           std::int64_t start_row = 0) {
  const std::size_t n_a = a.size() - window + 1;
  const std::size_t n_b = b.size() - window + 1;
  const std::size_t output_size = columns ? n_a : n_b;
  ExpectedProfile result{std::vector<float>(output_size, -2.0F),
                         std::vector<std::int32_t>(output_size, -1)};
  const std::size_t exclusion = (window + 3) / 4;

  for (std::size_t col = 0; col < n_a; ++col) {
    for (std::size_t row = 0; row < n_b; ++row) {
      const auto global_col = static_cast<std::int64_t>(col) + start_col;
      const auto global_row = static_cast<std::int64_t>(row) + start_row;
      if ((self_join || aligned) && std::abs(global_col - global_row) <
                                        static_cast<std::int64_t>(exclusion)) {
        continue;
      }
      double mean_a = 0.0;
      double mean_b = 0.0;
      bool valid = true;
      for (std::size_t k = 0; k < window; ++k) {
        valid = valid && std::isfinite(a[col + k]) && std::isfinite(b[row + k]);
        mean_a += a[col + k];
        mean_b += b[row + k];
      }
      if (!valid) {
        continue;
      }
      mean_a /= static_cast<double>(window);
      mean_b /= static_cast<double>(window);
      double covariance = 0.0;
      double norm_a = 0.0;
      double norm_b = 0.0;
      for (std::size_t k = 0; k < window; ++k) {
        const double centered_a = a[col + k] - mean_a;
        const double centered_b = b[row + k] - mean_b;
        covariance += centered_a * centered_b;
        norm_a += centered_a * centered_a;
        norm_b += centered_b * centered_b;
      }
      if (norm_a <= 1e-13 || norm_b <= 1e-13) {
        continue;
      }
      const float corr = static_cast<float>(
          std::clamp(covariance / std::sqrt(norm_a * norm_b), -1.0, 1.0));
      const std::size_t target = columns ? col : row;
      const std::int32_t match = static_cast<std::int32_t>(columns ? row : col);
      if (corr > result.correlation[target] + 1e-6F ||
          (std::abs(corr - result.correlation[target]) <= 1e-6F &&
           (result.index[target] < 0 || match < result.index[target]))) {
        result.correlation[target] = corr;
        result.index[target] = match;
      }
    }
  }
  return result;
}

void CheckProfile(const std::vector<std::uint64_t> &actual,
                  const ExpectedProfile &expected, float tolerance) {
  if (actual.size() != expected.correlation.size()) {
    throw std::runtime_error("profile size mismatch");
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float correlation = SCAMP::GetProfileCorrelation(actual[i]);
    const std::int32_t index = SCAMP::GetProfileIndex(actual[i]);
    if (index != expected.index[i] ||
        std::abs(correlation - expected.correlation[i]) > tolerance) {
      throw std::runtime_error("profile mismatch at " + std::to_string(i) +
                               ": got (" + std::to_string(correlation) + ", " +
                               std::to_string(index) + "), expected (" +
                               std::to_string(expected.correlation[i]) + ", " +
                               std::to_string(expected.index[i]) + ")");
    }
  }
}

void TestSelfJoin() {
  const std::vector<double> series = {
      0.2, 1.3, -0.7, 2.1, 0.5, -1.4, 0.9, 1.8,  -0.2, 0.4, 1.1,  -0.8,
      0.6, 1.5, -1.1, 0.3, 1.9, -0.4, 0.8, -1.6, 0.7,  1.2, -0.5, 1.7};
  SCAMP::SCAMPArgs args;
  args.timeseries_a = series;
  args.window = 5;
  SCAMP::do_SCAMP(&args);
  CheckProfile(args.profile_a.data.at(0).uint64_value,
               BruteForce(series, series, args.window, true, true), 3e-5F);
}

void TestABJoinBothDirections() {
  const std::vector<double> a = {0.1, 1.0,  -0.3, 0.8, 1.7,  -1.2, 0.4,
                                 1.1, -0.6, 0.2,  1.5, -0.9, 0.7,  1.3};
  const std::vector<double> b = {-0.8, 0.5, 1.4, -0.1, 0.9, -1.5, 0.3,  1.8,
                                 -0.4, 0.6, 1.2, -0.7, 0.0, 1.6,  -1.0, 0.8};
  SCAMP::SCAMPArgs args;
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.has_b = true;
  args.window = 4;
  args.computing_rows = true;
  args.keep_rows_separate = true;
  SCAMP::do_SCAMP(&args, {0}, 0);
  CheckProfile(args.profile_a.data.at(0).uint64_value,
               BruteForce(a, b, args.window, false, true), 3e-5F);
  CheckProfile(args.profile_b.data.at(0).uint64_value,
               BruteForce(a, b, args.window, false, false), 3e-5F);
}

void TestInvalidWindows() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> a = {0.0, 1.0, nan,  -1.0, 0.5, 1.5,
                                 0.2, 1.1, -0.4, 0.8,  1.7, -0.2};
  const std::vector<double> b = {1.0, -0.2, 0.4, 1.3, -0.7, 0.8,
                                 1.5, -1.0, 0.1, 0.6, 1.2,  -0.3};
  SCAMP::SCAMPArgs args;
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.has_b = true;
  args.window = 4;
  args.computing_rows = false;
  SCAMP::do_SCAMP(&args);
  CheckProfile(args.profile_a.data.at(0).uint64_value,
               BruteForce(a, b, args.window, false, true), 3e-5F);
}

void TestAlignedABJoinAndOffsets() {
  const std::vector<double> a = {0.1, 1.0,  -0.3, 0.8, 1.7,  -1.2, 0.4,
                                 1.1, -0.6, 0.2,  1.5, -0.9, 0.7,  1.3};
  const std::vector<double> b = {0.7, -0.1, 0.5,  1.4, -0.8, 0.2,  1.6, -1.1,
                                 0.9, 0.3,  -0.5, 1.2, 0.0,  -0.7, 1.8, 0.4};
  SCAMP::SCAMPArgs args;
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.has_b = true;
  args.window = 4;
  args.computing_rows = false;
  args.is_aligned = true;
  args.distributed_start_col = 3;
  args.distributed_start_row = 1;
  SCAMP::do_SCAMP(&args);
  CheckProfile(args.profile_a.data.at(0).uint64_value,
               BruteForce(a, b, args.window, false, true, true, 3, 1), 3e-5F);
}

void TestABRowOnly() {
  const std::vector<double> a = {0.4, -0.2, 1.1, 0.8, -1.0, 0.3,
                                 1.7, -0.5, 0.6, 1.2, -0.9, 0.1};
  const std::vector<double> b = {-0.7, 0.5,  1.3, -0.1, 0.9,  -1.4, 0.2,
                                 1.6,  -0.3, 0.8, 1.0,  -0.6, 0.4};
  SCAMP::SCAMPArgs args;
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.has_b = true;
  args.window = 3;
  args.computing_columns = false;
  args.computing_rows = true;
  args.keep_rows_separate = true;
  SCAMP::do_SCAMP(&args);
  CheckProfile(args.profile_b.data.at(0).uint64_value,
               BruteForce(a, b, args.window, false, false), 4e-5F);
}

void TestHighOffsetInputPreservesVariation() {
  const double baseline = 1.0e12;
  const std::vector<double> a = {
      baseline + 0.0, baseline + 1.0, baseline + 4.0,  baseline + 2.0,
      baseline + 8.0, baseline + 3.0, baseline + 7.0,  baseline + 5.0,
      baseline + 9.0, baseline + 6.0, baseline + 11.0, baseline + 10.0};
  const std::vector<double> b = {
      baseline + 6.0, baseline + 2.0, baseline + 9.0,  baseline + 1.0,
      baseline + 7.0, baseline + 4.0, baseline + 10.0, baseline + 3.0,
      baseline + 8.0, baseline + 0.0, baseline + 11.0, baseline + 5.0,
      baseline + 12.0};
  SCAMP::SCAMPArgs args;
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.has_b = true;
  args.window = 4;
  args.computing_rows = false;
  SCAMP::do_SCAMP(&args);
  CheckProfile(args.profile_a.data.at(0).uint64_value,
               BruteForce(a, b, args.window, false, true), 5e-5F);
}

void TestRandomizedJoins() {
  std::uint32_t state = 0x9e3779b9U;
  auto sample = [&state]() {
    state = state * 1664525U + 1013904223U;
    return (static_cast<double>((state >> 8U) & 0x00FFFFFFU) /
                static_cast<double>(0x01000000U) -
            0.5) *
           4.0;
  };

  for (int trial = 0; trial < 8; ++trial) {
    const std::size_t window = static_cast<std::size_t>(3 + trial % 5);
    std::vector<double> a(static_cast<std::size_t>(25 + trial));
    std::vector<double> b(static_cast<std::size_t>(28 + 2 * trial));
    std::generate(a.begin(), a.end(), sample);
    std::generate(b.begin(), b.end(), sample);

    SCAMP::SCAMPArgs self_args;
    self_args.timeseries_a = a;
    self_args.window = window;
    SCAMP::do_SCAMP(&self_args);
    CheckProfile(self_args.profile_a.data.at(0).uint64_value,
                 BruteForce(a, a, window, true, true), 5e-5F);

    SCAMP::SCAMPArgs ab_args;
    ab_args.timeseries_a = a;
    ab_args.timeseries_b = b;
    ab_args.has_b = true;
    ab_args.window = window;
    ab_args.computing_rows = false;
    SCAMP::do_SCAMP(&ab_args);
    CheckProfile(ab_args.profile_a.data.at(0).uint64_value,
                 BruteForce(a, b, window, false, true), 5e-5F);
  }
}

void TestUnsupportedModeIsExplicit() {
  SCAMP::SCAMPArgs args;
  args.timeseries_a.assign(32, 1.0);
  args.window = 4;
  args.precision_type = SCAMP::PRECISION_DOUBLE;
  try {
    SCAMP::do_SCAMP(&args);
  } catch (const SCAMP::SCAMPException &error) {
    if (std::string(error.what()).find("PRECISION_SINGLE") !=
        std::string::npos) {
      return;
    }
    throw;
  }
  throw std::runtime_error("unsupported precision did not throw");
}

void TestShortSelfJoinHasNoNontrivialMatch() {
  SCAMP::SCAMPArgs args;
  args.timeseries_a = {0.0, 1.0, -1.0};
  args.window = 3;
  SCAMP::do_SCAMP(&args);
  const auto packed = args.profile_a.data.at(0).uint64_value.at(0);
  if (SCAMP::GetProfileCorrelation(packed) != -2.0F ||
      SCAMP::GetProfileIndex(packed) != -1) {
    throw std::runtime_error("short self join did not retain invalid sentinel");
  }
}

} // namespace

int main() {
  try {
    TestSelfJoin();
    TestABJoinBothDirections();
    TestInvalidWindows();
    TestAlignedABJoinAndOffsets();
    TestABRowOnly();
    TestHighOffsetInputPreservesVariation();
    TestRandomizedJoins();
    TestUnsupportedModeIsExplicit();
    TestShortSelfJoinHasNoNontrivialMatch();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "native indexed 1NN tests passed\n";
  return 0;
}
