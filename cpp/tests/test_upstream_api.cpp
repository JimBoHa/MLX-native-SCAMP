#include <common/common.h>
#include <common/profile.h>
#include <common/scamp_args.h>
#include <common/scamp_exception.h>
#include <common/scamp_interface.h>

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

static_assert(std::is_same_v<SCAMPException, SCAMP::SCAMPException>);
static_assert(std::is_base_of_v<std::exception, SCAMPException>);
static_assert(sizeof(SCAMP::mp_entry) == sizeof(std::uint64_t));
static_assert(std::is_same_v<decltype(&SCAMP::SCAMPArgs::validate),
                             void (SCAMP::SCAMPArgs::*)()>);
static_assert(std::is_same_v<decltype(&SCAMP::SCAMPArgs::print),
                             void (SCAMP::SCAMPArgs::*)()>);

int main() {
  if (SCAMP::GetPrecisionTypeString(SCAMP::PRECISION_SINGLE) !=
          "PRECISION_SINGLE" ||
      SCAMP::GetProfileTypeString(SCAMP::PROFILE_TYPE_1NN_INDEX) !=
          "PROFILE_TYPE_1NN_INDEX") {
    return 1;
  }
  SCAMP::SCAMPArgs args;
  args.max_matches_per_column = 17;
  std::ostringstream printed;
  std::streambuf *original = std::cout.rdbuf(printed.rdbuf());
  args.print();
  std::cout.rdbuf(original);
  if (printed.str().find("max_matches_per_column: 17\n") ==
      std::string::npos) {
    return 3;
  }
  try {
    throw SCAMPException("upstream-compatible exception");
  } catch (const SCAMPException &error) {
    return std::string(error.what()) == "upstream-compatible exception" ? 0 : 2;
  }
}
