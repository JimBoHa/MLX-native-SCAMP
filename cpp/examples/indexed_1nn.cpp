#include <common/scamp_interface.h>

#include <cmath>
#include <iostream>

int main() {
  SCAMP::SCAMPArgs args;
  args.timeseries_a = {0.0, 1.0,  0.0, -1.0, 0.0, 1.0,
                       0.0, -1.0, 0.0, 1.0,  0.0, -1.0};
  args.window = 4;
  args.silent_mode = false;

  SCAMP::do_SCAMP(&args);

  const auto &profile = args.profile_a.data.at(0).uint64_value;
  for (std::size_t i = 0; i < profile.size(); ++i) {
    std::cout << i
              << ": correlation=" << SCAMP::GetProfileCorrelation(profile[i])
              << ", index=" << SCAMP::GetProfileIndex(profile[i]) << '\n';
  }
  return 0;
}
