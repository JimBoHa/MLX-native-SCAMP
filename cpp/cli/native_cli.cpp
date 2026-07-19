#include "native_cli.h"

#include <scamp/scamp.h>

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <locale.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace mlx_scamp::cli {
namespace {

namespace fs = std::filesystem;

class CliError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class LocaleHandle {
public:
  LocaleHandle() : value_(newlocale(LC_NUMERIC_MASK, "C", nullptr)) {
    if (value_ == nullptr) {
      throw CliError("unable to create a locale-independent number parser");
    }
  }
  LocaleHandle(const LocaleHandle &) = delete;
  LocaleHandle &operator=(const LocaleHandle &) = delete;
  ~LocaleHandle() { freelocale(value_); }

  locale_t get() const { return value_; }

private:
  locale_t value_;
};

struct Options {
  std::int64_t window{-1};
  std::string input_a;
  std::string input_b;
  std::string output_a{"mp_columns_out"};
  std::string output_a_index{"mp_columns_out_index"};
  std::string output_b{"mp_rows_out"};
  std::string output_b_index{"mp_rows_out_index"};
  std::string profile_type{"1NN_INDEX"};
  std::string gpus;
  std::int64_t global_row{-1};
  std::int64_t global_col{-1};
  std::int64_t num_cpu_workers{0};
  std::int64_t max_tile_size{1 << 17};
  std::int64_t max_matches_per_column{5};
  std::int64_t reduced_height{50};
  std::int64_t reduced_width{50};
  double threshold{0.0};
  bool single_precision{false};
  bool double_precision{false};
  bool ultra_precision{false};
  bool output_pearson{false};
  bool print_debug_info{false};
  bool keep_rows{false};
  bool aligned{false};
  bool no_gpu{false};
  bool autotune{false};
  bool list_variants{false};
  bool help{false};
  std::unordered_set<std::string> explicitly_set;
};

constexpr std::string_view kVariant =
    "v0 backend=mlx-metal strategy=two-pass-diagonal-recurrence "
    "profile=1NN_INDEX precision=single device=0";

const std::unordered_set<std::string> &BoolFlags() {
  static const std::unordered_set<std::string> flags = {"single_precision",
                                                        "double_precision",
                                                        "ultra_precision",
                                                        "output_pearson",
                                                        "print_debug_info",
                                                        "keep_rows",
                                                        "aligned",
                                                        "no_gpu",
                                                        "autotune",
                                                        "list_variants",
                                                        "help",
                                                        "helpshort",
                                                        "helpfull"};
  return flags;
}

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](char ch) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  });
  return value;
}

bool IsBooleanValue(std::string_view text) {
  const std::string value = Lower(std::string(text));
  return value == "true" || value == "false" || value == "1" || value == "0" ||
         value == "t" || value == "f" || value == "yes" || value == "no" ||
         value == "y" || value == "n";
}

bool ParseBoolean(std::string_view text, std::string_view flag) {
  const std::string value = Lower(std::string(text));
  if (value == "true" || value == "1" || value == "t" || value == "yes" ||
      value == "y") {
    return true;
  }
  if (value == "false" || value == "0" || value == "f" || value == "no" ||
      value == "n") {
    return false;
  }
  throw CliError("invalid boolean value for --" + std::string(flag) + ": " +
                 std::string(text));
}

std::int64_t ParseInteger(std::string_view text, std::string_view flag) {
  if (text.empty()) {
    throw CliError("missing integer value for --" + std::string(flag));
  }
  bool positive_sign = text.front() == '+';
  if (positive_sign) {
    text.remove_prefix(1);
  }
  if (text.empty()) {
    throw CliError("invalid integer value for --" + std::string(flag));
  }
  std::int64_t value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (result.ec == std::errc::result_out_of_range) {
    throw CliError("integer value for --" + std::string(flag) +
                   " is out of range");
  }
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw CliError("invalid integer value for --" + std::string(flag) + ": " +
                   std::string(text));
  }
  return value;
}

double ParseDouble(std::string_view text, std::string_view flag) {
  if (text.empty()) {
    throw CliError("missing numeric value for --" + std::string(flag));
  }
  std::string owned(text);
  LocaleHandle c_locale;
  errno = 0;
  char *end = nullptr;
  const double value = strtod_l(owned.c_str(), &end, c_locale.get());
  const int parse_errno = errno;
  if (end != owned.c_str() + owned.size() || end == owned.c_str()) {
    throw CliError("invalid numeric value for --" + std::string(flag) + ": " +
                   owned);
  }
  if (parse_errno == ERANGE) {
    throw CliError("numeric value for --" + std::string(flag) +
                   " is out of range");
  }
  return value;
}

void SetBoolean(Options *options, const std::string &name, bool value) {
  if (name == "single_precision") {
    options->single_precision = value;
  } else if (name == "double_precision") {
    options->double_precision = value;
  } else if (name == "ultra_precision") {
    options->ultra_precision = value;
  } else if (name == "output_pearson") {
    options->output_pearson = value;
  } else if (name == "print_debug_info") {
    options->print_debug_info = value;
  } else if (name == "keep_rows") {
    options->keep_rows = value;
  } else if (name == "aligned") {
    options->aligned = value;
  } else if (name == "no_gpu") {
    options->no_gpu = value;
  } else if (name == "autotune") {
    options->autotune = value;
  } else if (name == "list_variants") {
    options->list_variants = value;
  } else if (name == "help" || name == "helpshort" || name == "helpfull") {
    options->help = value;
  } else {
    throw CliError("unknown flag --" + name);
  }
}

void SetValue(Options *options, const std::string &name,
              std::string_view value) {
  if (name == "window") {
    options->window = ParseInteger(value, name);
  } else if (name == "input_a_file_name") {
    options->input_a = value;
  } else if (name == "input_b_file_name") {
    options->input_b = value;
  } else if (name == "output_a_file_name") {
    options->output_a = value;
  } else if (name == "output_a_index_file_name") {
    options->output_a_index = value;
  } else if (name == "output_b_file_name") {
    options->output_b = value;
  } else if (name == "output_b_index_file_name") {
    options->output_b_index = value;
  } else if (name == "profile_type") {
    options->profile_type = value;
  } else if (name == "gpus") {
    options->gpus = value;
  } else if (name == "global_row") {
    options->global_row = ParseInteger(value, name);
  } else if (name == "global_col") {
    options->global_col = ParseInteger(value, name);
  } else if (name == "num_cpu_workers") {
    options->num_cpu_workers = ParseInteger(value, name);
  } else if (name == "max_tile_size") {
    options->max_tile_size = ParseInteger(value, name);
  } else if (name == "max_matches_per_column") {
    options->max_matches_per_column = ParseInteger(value, name);
  } else if (name == "reduced_height") {
    options->reduced_height = ParseInteger(value, name);
  } else if (name == "reduced_width") {
    options->reduced_width = ParseInteger(value, name);
  } else if (name == "threshold") {
    options->threshold = ParseDouble(value, name);
  } else {
    throw CliError("unknown flag --" + name);
  }
}

Options ParseOptions(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string argument(argv[i]);
    if (argument == "-h") {
      options.help = true;
      options.explicitly_set.insert("help");
      continue;
    }
    if (argument.size() < 2 || argument.front() != '-') {
      throw CliError("unexpected positional argument: " + argument);
    }
    std::size_t prefix = argument.rfind("--", 0) == 0 ? 2 : 1;
    std::string body = argument.substr(prefix);
    if (body.empty()) {
      throw CliError("invalid empty flag");
    }
    const std::size_t equals = body.find('=');
    std::string name = body.substr(0, equals);
    std::optional<std::string> inline_value;
    if (equals != std::string::npos) {
      inline_value = body.substr(equals + 1);
    }

    bool negated = false;
    if (BoolFlags().count(name) == 0 && name.rfind("no", 0) == 0 &&
        BoolFlags().count(name.substr(2)) != 0) {
      negated = true;
      name = name.substr(2);
    }

    if (BoolFlags().count(name) != 0) {
      if (negated && inline_value.has_value()) {
        throw CliError("negated boolean --no" + name +
                       " cannot also have a value");
      }
      bool value = !negated;
      if (inline_value.has_value()) {
        value = ParseBoolean(*inline_value, name);
      } else if (!negated && i + 1 < argc && IsBooleanValue(argv[i + 1])) {
        value = ParseBoolean(argv[++i], name);
      }
      SetBoolean(&options, name, value);
      options.explicitly_set.insert(name);
      continue;
    }
    if (negated) {
      throw CliError("unknown flag --no" + name);
    }

    std::string value;
    if (inline_value.has_value()) {
      value = *inline_value;
    } else {
      if (i + 1 >= argc) {
        throw CliError("missing value for --" + name);
      }
      value = argv[++i];
    }
    SetValue(&options, name, value);
    options.explicitly_set.insert(name);
  }
  return options;
}

void PrintHelp() {
  std::cout
      << "mlx-scamp-native: native Apple Silicon indexed-1NN SCAMP CLI\n\n"
      << "Required job flags:\n"
      << "  --window=N\n"
      << "  --input_a_file_name=PATH\n"
      << "  --single_precision\n\n"
      << "Join/output flags:\n"
      << "  --input_b_file_name=PATH       Perform an AB join\n"
      << "  --keep_rows                    Write the AB row profile too\n"
      << "  --output_a_file_name=PATH      Default: mp_columns_out\n"
      << "  --output_a_index_file_name=PATH\n"
      << "  --output_b_file_name=PATH      Default: mp_rows_out\n"
      << "  --output_b_index_file_name=PATH\n"
      << "  --output_pearson               Default output is z-normalized "
         "Euclidean distance\n"
      << "  --aligned --global_row=N --global_col=N\n"
      << "  --gpus=0                       MLX exposes Metal device 0\n"
      << "  --print_debug_info\n\n"
      << "Only --profile_type=1NN_INDEX and single precision are currently "
         "implemented.\n"
      << "Flags accept --name=value or --name value. Boolean flags also "
         "accept\n"
      << "--flag=true|false and --noflag. Use --list_variants for the fixed "
         "Metal strategy.\n";
}

void ValidateCapabilities(const Options &options) {
  if (options.profile_type != "1NN_INDEX") {
    throw CliError("--profile_type=" + options.profile_type +
                   " is unsupported; this native CLI currently implements "
                   "1NN_INDEX only");
  }
  if (!options.single_precision) {
    throw CliError("--single_precision is required by the native Metal "
                   "indexed-1NN kernel");
  }
  if (options.double_precision || options.ultra_precision) {
    throw CliError("double and ultra precision are not implemented by this "
                   "native Metal CLI");
  }
  if (options.no_gpu) {
    throw CliError("--no_gpu is unsupported because a native CPU path is not "
                   "implemented");
  }
  if (options.num_cpu_workers != 0) {
    throw CliError("--num_cpu_workers must be 0; native CPU workers are not "
                   "implemented");
  }
  if (options.autotune) {
    throw CliError("--autotune is unsupported; the MLX Metal kernel uses the "
                   "fixed strategy reported by --list_variants");
  }
  if (options.explicitly_set.count("max_tile_size") != 0) {
    throw CliError("explicit --max_tile_size is unsupported because this "
                   "native kernel does not yet tile a join");
  }
  for (const std::string &name : {"threshold", "max_matches_per_column",
                                  "reduced_height", "reduced_width"}) {
    if (options.explicitly_set.count(name) != 0) {
      throw CliError("--" + name +
                     " applies to a reducer not implemented by this "
                     "indexed-1NN CLI");
    }
  }
  if (!options.gpus.empty() && options.gpus != "0") {
    throw CliError("--gpus must be empty or 0 on Apple Silicon MLX");
  }
  if (options.window < 3) {
    throw CliError("--window must be an integer of at least 3");
  }
  if (options.window > 256000) {
    throw CliError("--window exceeds the current native join-wide dispatch "
                   "limit of 256000");
  }
  if (options.input_a.empty()) {
    throw CliError("primary input filename must be specified with "
                   "--input_a_file_name");
  }
  const bool has_b = !options.input_b.empty();
  if (options.output_a.empty() || options.output_a_index.empty() ||
      (options.keep_rows &&
       (options.output_b.empty() || options.output_b_index.empty()))) {
    throw CliError("active output file names must not be empty");
  }
  if (options.keep_rows && !has_b) {
    throw CliError("--keep_rows is valid only for an AB join");
  }
  const bool row_set = options.global_row != -1;
  const bool col_set = options.global_col != -1;
  if (row_set != col_set) {
    throw CliError("--global_row and --global_col must be supplied together");
  }
  if (options.global_row < -1 || options.global_col < -1) {
    throw CliError("global offsets must be -1 (unset) or nonnegative");
  }
  if (row_set && (!has_b || !options.aligned)) {
    throw CliError("global offsets require an aligned AB join");
  }
  if (options.global_row > std::numeric_limits<std::int32_t>::max() ||
      options.global_col > std::numeric_limits<std::int32_t>::max()) {
    throw CliError("global offsets exceed the Metal int32 index range");
  }
}

std::string FoldPath(const fs::path &path) {
  const std::string utf8 = path.string();
  CFStringRef source = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(utf8.data()),
      static_cast<CFIndex>(utf8.size()), kCFStringEncodingUTF8, false);
  if (source == nullptr) {
    throw CliError("path is not valid UTF-8: " + utf8);
  }
  CFMutableStringRef folded =
      CFStringCreateMutableCopy(kCFAllocatorDefault, 0, source);
  CFRelease(source);
  if (folded == nullptr) {
    throw CliError("unable to normalize path: " + utf8);
  }
  CFStringNormalize(folded, kCFStringNormalizationFormKC);
  CFStringFold(folded,
               kCFCompareCaseInsensitive | kCFCompareWidthInsensitive |
                   kCFCompareNonliteral,
               nullptr);
  const CFIndex capacity =
      CFStringGetMaximumSizeForEncoding(CFStringGetLength(folded),
                                        kCFStringEncodingUTF8) +
      1;
  std::vector<char> output(static_cast<std::size_t>(capacity));
  const bool converted = CFStringGetCString(folded, output.data(), capacity,
                                            kCFStringEncodingUTF8);
  CFRelease(folded);
  if (!converted) {
    throw CliError("unable to normalize path: " + utf8);
  }
  return output.data();
}

fs::path ResolvedPath(const fs::path &path) {
  std::error_code error;
  fs::path absolute = fs::absolute(path, error);
  if (error) {
    throw CliError("unable to resolve path " + path.string() + ": " +
                   error.message());
  }
  fs::path resolved = fs::weakly_canonical(absolute, error);
  if (error) {
    // The nearest existing parent is validated separately. Keeping the
    // lexical absolute form here still lets conservative case/Unicode
    // comparison protect a not-yet-created output.
    resolved = absolute.lexically_normal();
  }
  return resolved;
}

struct PathIdentity {
  std::string folded;
  std::optional<std::pair<dev_t, ino_t>> physical;
};

PathIdentity Identity(const fs::path &path) {
  PathIdentity result{FoldPath(ResolvedPath(path)), std::nullopt};
  struct stat status{};
  if (::stat(path.c_str(), &status) == 0) {
    result.physical = std::make_pair(status.st_dev, status.st_ino);
  } else if (errno != ENOENT && errno != ENOTDIR) {
    throw CliError("unable to inspect path " + path.string() + ": " +
                   std::strerror(errno));
  }
  return result;
}

bool SameIdentity(const PathIdentity &left, const PathIdentity &right) {
  return left.folded == right.folded ||
         (left.physical.has_value() && right.physical.has_value() &&
          left.physical == right.physical);
}

bool PathExistsWithoutFollowing(const fs::path &path) {
  struct stat status{};
  if (::lstat(path.c_str(), &status) == 0) {
    return true;
  }
  if (errno == ENOENT || errno == ENOTDIR) {
    return false;
  }
  throw CliError("unable to inspect path " + path.string() + ": " +
                 std::strerror(errno));
}

void ValidateInput(const fs::path &path) {
  std::error_code error;
  const fs::file_status status = fs::status(path, error);
  if (error || !fs::exists(status)) {
    throw CliError("unable to open " + path.string() + " for reading");
  }
  if (!fs::is_regular_file(status)) {
    throw CliError("input is not a regular file: " + path.string());
  }
}

void ValidateOutput(const fs::path &path) {
  fs::path parent = path.parent_path();
  if (parent.empty()) {
    parent = ".";
  }
  std::error_code error;
  const fs::file_status parent_status = fs::status(parent, error);
  if (error || !fs::is_directory(parent_status)) {
    throw CliError("output directory does not exist: " + parent.string());
  }
  const fs::file_status link_status = fs::symlink_status(path, error);
  if (error && error != std::errc::no_such_file_or_directory) {
    throw CliError("unable to inspect output " + path.string() + ": " +
                   error.message());
  }
  if (!error && fs::is_symlink(link_status)) {
    throw CliError("output symlinks are not supported; name the intended "
                   "target directly: " +
                   path.string());
  } else if (!error && fs::exists(link_status) &&
             !fs::is_regular_file(link_status)) {
    throw CliError("output is not a regular file: " + path.string());
  }
}

std::vector<fs::path> ActiveOutputs(const Options &options) {
  std::vector<fs::path> outputs = {options.output_a, options.output_a_index};
  if (options.keep_rows) {
    outputs.emplace_back(options.output_b);
    outputs.emplace_back(options.output_b_index);
  }
  return outputs;
}

void ValidatePathSafety(const Options &options) {
  std::vector<fs::path> inputs = {options.input_a};
  if (!options.input_b.empty()) {
    inputs.emplace_back(options.input_b);
  }
  const std::vector<fs::path> outputs = ActiveOutputs(options);
  for (const fs::path &input : inputs) {
    ValidateInput(input);
  }
  for (const fs::path &output : outputs) {
    ValidateOutput(output);
  }

  std::vector<PathIdentity> input_ids;
  std::vector<PathIdentity> output_ids;
  input_ids.reserve(inputs.size());
  output_ids.reserve(outputs.size());
  for (const fs::path &input : inputs) {
    input_ids.push_back(Identity(input));
  }
  for (const fs::path &output : outputs) {
    output_ids.push_back(Identity(output));
  }
  for (std::size_t output = 0; output < outputs.size(); ++output) {
    for (std::size_t input = 0; input < inputs.size(); ++input) {
      if (SameIdentity(output_ids[output], input_ids[input])) {
        throw CliError(
            "output path aliases input path: " + outputs[output].string() +
            " and " + inputs[input].string());
      }
    }
    for (std::size_t earlier = 0; earlier < output; ++earlier) {
      if (SameIdentity(output_ids[output], output_ids[earlier])) {
        throw CliError(
            "output paths alias each other: " + outputs[earlier].string() +
            " and " + outputs[output].string());
      }
    }
  }
}

std::vector<double> ReadSeries(const fs::path &path) {
  std::ifstream input(path);
  input.imbue(std::locale::classic());
  if (!input) {
    throw CliError("unable to open " + path.string() + " for reading");
  }
  LocaleHandle c_locale;
  std::vector<double> values;
  std::string token;
  std::size_t number = 0;
  while (input >> token) {
    ++number;
    errno = 0;
    char *end = nullptr;
    const double value = strtod_l(token.c_str(), &end, c_locale.get());
    if (end == token.c_str() || end != token.c_str() + token.size()) {
      throw CliError("could not parse token " + std::to_string(number) +
                     " in " + path.string() + ": " + token);
    }
    if (errno == ERANGE) {
      throw CliError("token " + std::to_string(number) + " in " +
                     path.string() + " is out of range");
    }
    values.push_back(value);
  }
  if (input.bad()) {
    throw CliError("I/O error while reading " + path.string());
  }
  return values;
}

class FileDescriptorBuffer : public std::streambuf {
public:
  explicit FileDescriptorBuffer(int descriptor) : descriptor_(descriptor) {
    setp(buffer_.data(), buffer_.data() + buffer_.size());
  }

  bool Flush() {
    const std::ptrdiff_t count = pptr() - pbase();
    char *cursor = pbase();
    std::ptrdiff_t remaining = count;
    while (remaining > 0) {
      const ssize_t written =
          ::write(descriptor_, cursor, static_cast<std::size_t>(remaining));
      if (written < 0) {
        if (errno == EINTR) {
          continue;
        }
        failed_ = true;
        return false;
      }
      if (written == 0) {
        failed_ = true;
        return false;
      }
      cursor += written;
      remaining -= written;
    }
    setp(buffer_.data(), buffer_.data() + buffer_.size());
    return !failed_;
  }

protected:
  int_type overflow(int_type character) override {
    if (!Flush()) {
      return traits_type::eof();
    }
    if (!traits_type::eq_int_type(character, traits_type::eof())) {
      *pptr() = traits_type::to_char_type(character);
      pbump(1);
    }
    return traits_type::not_eof(character);
  }

  int sync() override { return Flush() ? 0 : -1; }

private:
  int descriptor_;
  std::array<char, 64 * 1024> buffer_{};
  bool failed_{false};
};

struct TemporaryFile {
  fs::path path;
  int descriptor{-1};
};

TemporaryFile CreateTemporaryFile(const fs::path &target,
                                  std::string_view purpose) {
  fs::path parent = target.parent_path();
  if (parent.empty()) {
    parent = ".";
  }
  std::string pattern =
      (parent / (".mlx-scamp-" + std::string(purpose) + "-XXXXXX")).string();
  std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
  mutable_pattern.push_back('\0');
  const int descriptor = ::mkstemp(mutable_pattern.data());
  if (descriptor < 0) {
    throw CliError("unable to create temporary output beside " +
                   target.string() + ": " + std::strerror(errno));
  }
  return {fs::path(mutable_pattern.data()), descriptor};
}

mode_t DesiredOutputMode(const fs::path &target) {
  struct stat status{};
  if (::stat(target.c_str(), &status) == 0) {
    return status.st_mode & 0777;
  }
  const mode_t mask = ::umask(0);
  ::umask(mask);
  return static_cast<mode_t>(0666 & ~mask);
}

struct StagedOutput {
  fs::path target;
  fs::path temporary;
  fs::path backup;
  bool temporary_exists{true};
  bool had_target{false};
  bool backup_exists{false};
  bool installed{false};
};

template <typename Writer>
StagedOutput StageOutput(const fs::path &target, Writer writer) {
  TemporaryFile temporary = CreateTemporaryFile(target, "output");
  try {
    if (::fchmod(temporary.descriptor, DesiredOutputMode(target)) != 0) {
      throw CliError("unable to set permissions on temporary output: " +
                     std::string(std::strerror(errno)));
    }
    FileDescriptorBuffer buffer(temporary.descriptor);
    std::ostream output(&buffer);
    output.imbue(std::locale::classic());
    output << std::setprecision(10);
    writer(output);
    output.flush();
    if (!output || !buffer.Flush()) {
      throw CliError("unable to write temporary output for " + target.string());
    }
    if (::fsync(temporary.descriptor) != 0) {
      throw CliError("unable to fsync temporary output for " + target.string() +
                     ": " + std::strerror(errno));
    }
    if (::close(temporary.descriptor) != 0) {
      temporary.descriptor = -1;
      throw CliError("unable to close temporary output for " + target.string() +
                     ": " + std::strerror(errno));
    }
    temporary.descriptor = -1;
    return {target, temporary.path, fs::path{}, true, false, false, false};
  } catch (...) {
    if (temporary.descriptor >= 0) {
      ::close(temporary.descriptor);
    }
    ::unlink(temporary.path.c_str());
    throw;
  }
}

void FsyncDirectories(const std::vector<StagedOutput> &outputs) {
  std::set<fs::path> parents;
  for (const StagedOutput &output : outputs) {
    fs::path parent = output.target.parent_path();
    if (parent.empty()) {
      parent = ".";
    }
    parents.insert(ResolvedPath(parent));
  }
  for (const fs::path &parent : parents) {
    const int descriptor = ::open(parent.c_str(), O_RDONLY);
    if (descriptor < 0) {
      throw CliError("unable to open output directory for fsync: " +
                     parent.string());
    }
    const int result = ::fsync(descriptor);
    const int saved_errno = errno;
    ::close(descriptor);
    if (result != 0) {
      throw CliError("unable to fsync output directory " + parent.string() +
                     ": " + std::strerror(saved_errno));
    }
  }
}

void RollBack(std::vector<StagedOutput> *outputs) noexcept {
  for (auto iterator = outputs->rbegin(); iterator != outputs->rend();
       ++iterator) {
    StagedOutput &output = *iterator;
    if (output.had_target && output.backup_exists) {
      if (::rename(output.backup.c_str(), output.target.c_str()) == 0) {
        output.backup_exists = false;
        output.installed = false;
      } else {
        std::cerr << "mlx-scamp-native: warning: rollback could not restore "
                  << output.target << " from " << output.backup << ": "
                  << std::strerror(errno) << '\n';
      }
    } else if (!output.had_target && output.installed) {
      if (::unlink(output.target.c_str()) == 0 || errno == ENOENT) {
        output.installed = false;
      } else {
        std::cerr << "mlx-scamp-native: warning: rollback could not remove "
                  << output.target << ": " << std::strerror(errno) << '\n';
      }
    }
    if (output.temporary_exists) {
      if (::unlink(output.temporary.c_str()) == 0 || errno == ENOENT) {
        output.temporary_exists = false;
      } else {
        std::cerr << "mlx-scamp-native: warning: rollback could not remove "
                  << output.temporary << ": " << std::strerror(errno) << '\n';
      }
    }
  }
}

void CommitOutputs(std::vector<StagedOutput> *outputs) {
  try {
    for (StagedOutput &output : *outputs) {
      output.had_target = PathExistsWithoutFollowing(output.target);
      if (!output.had_target) {
        continue;
      }
      TemporaryFile backup = CreateTemporaryFile(output.target, "backup");
      if (::close(backup.descriptor) != 0) {
        ::unlink(backup.path.c_str());
        throw CliError("unable to close output backup placeholder");
      }
      output.backup = backup.path;
      if (::rename(output.target.c_str(), output.backup.c_str()) != 0) {
        ::unlink(output.backup.c_str());
        throw CliError("unable to stage existing output " +
                       output.target.string() + ": " + std::strerror(errno));
      }
      output.backup_exists = true;
    }
    for (StagedOutput &output : *outputs) {
      if (::rename(output.temporary.c_str(), output.target.c_str()) != 0) {
        throw CliError("unable to atomically install output " +
                       output.target.string() + ": " + std::strerror(errno));
      }
      output.temporary_exists = false;
      output.installed = true;
    }
    FsyncDirectories(*outputs);
  } catch (...) {
    RollBack(outputs);
    throw;
  }

  for (StagedOutput &output : *outputs) {
    if (output.backup_exists) {
      if (::unlink(output.backup.c_str()) == 0) {
        output.backup_exists = false;
      } else {
        std::cerr << "mlx-scamp-native: warning: committed output but could "
                     "not remove backup "
                  << output.backup << ": " << std::strerror(errno) << '\n';
      }
    }
  }
  // Persist removal of transaction backups as well as the installed names.
  try {
    FsyncDirectories(*outputs);
  } catch (const std::exception &error) {
    // The installed names were already fsynced while rollback backups still
    // existed. A failure to persist backup cleanup is actionable, but it must
    // not turn a completed multi-output commit into an apparent job failure.
    std::cerr << "mlx-scamp-native: warning: " << error.what() << '\n';
  }
}

double OutputValue(float correlation, bool pearson, std::uint64_t window) {
  if (correlation < -1.0F) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (pearson) {
    return correlation;
  }
  return std::sqrt(
      std::max(2.0 * static_cast<double>(window) * (1.0 - correlation), 0.0));
}

void StageProfile(const fs::path &value_path, const fs::path &index_path,
                  const std::vector<std::uint64_t> &profile, bool pearson,
                  std::uint64_t window, std::vector<StagedOutput> *staged) {
  staged->push_back(StageOutput(value_path, [&](std::ostream &output) {
    for (std::uint64_t packed : profile) {
      output << OutputValue(SCAMP::GetProfileCorrelation(packed), pearson,
                            window)
             << '\n';
    }
  }));
  staged->push_back(StageOutput(index_path, [&](std::ostream &output) {
    for (std::uint64_t packed : profile) {
      const float correlation = SCAMP::GetProfileCorrelation(packed);
      output << (correlation < -1.0F ? -1 : SCAMP::GetProfileIndex(packed))
             << '\n';
    }
  }));
}

void CleanStaged(std::vector<StagedOutput> *staged) noexcept {
  for (StagedOutput &output : *staged) {
    if (output.temporary_exists) {
      if (::unlink(output.temporary.c_str()) == 0 || errno == ENOENT) {
        output.temporary_exists = false;
      } else {
        std::cerr << "mlx-scamp-native: warning: could not remove staged file "
                  << output.temporary << ": " << std::strerror(errno) << '\n';
      }
    }
    if (output.backup_exists) {
      if (::rename(output.backup.c_str(), output.target.c_str()) == 0) {
        output.backup_exists = false;
        output.installed = false;
      } else {
        std::cerr << "mlx-scamp-native: warning: could not restore backup "
                  << output.backup << " to " << output.target << ": "
                  << std::strerror(errno) << '\n';
      }
    }
  }
}

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int Run(int argc, char **argv) {
  const Options options = ParseOptions(argc, argv);
  if (options.help) {
    PrintHelp();
    return 0;
  }
  if (options.list_variants) {
    std::cout << kVariant << '\n';
    return 0;
  }

  // Capability and path checks intentionally precede reading either input or
  // creating any output. An unsupported request can never truncate a user's
  // time series or replace an existing result.
  ValidateCapabilities(options);
  ValidatePathSafety(options);

  const auto start = std::chrono::steady_clock::now();
  std::cerr << "mlx-scamp-native: reading input\n";
  std::vector<double> series_a = ReadSeries(options.input_a);
  std::vector<double> series_b;
  if (!options.input_b.empty()) {
    series_b = ReadSeries(options.input_b);
  }

  SCAMP::SCAMPArgs arguments;
  arguments.timeseries_a = std::move(series_a);
  arguments.timeseries_b = std::move(series_b);
  arguments.has_b = !options.input_b.empty();
  arguments.window = static_cast<std::uint64_t>(options.window);
  arguments.max_tile_size = 512000;
  arguments.distributed_start_row = options.global_row;
  arguments.distributed_start_col = options.global_col;
  arguments.precision_type = SCAMP::PRECISION_SINGLE;
  arguments.profile_type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  arguments.profile_a.type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  arguments.profile_b.type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  arguments.computing_columns = true;
  arguments.computing_rows = !arguments.has_b || options.keep_rows;
  arguments.keep_rows_separate = options.keep_rows;
  arguments.is_aligned = options.aligned;
  arguments.silent_mode = true;

  if (options.print_debug_info) {
    std::cerr << "mlx-scamp-native: "
              << (arguments.has_b ? "AB join" : "self join")
              << ", window=" << arguments.window
              << ", samples_a=" << arguments.timeseries_a.size();
    if (arguments.has_b) {
      std::cerr << ", samples_b=" << arguments.timeseries_b.size();
    }
    std::cerr << ", output="
              << (options.output_pearson ? "pearson" : "euclidean") << '\n';
  }
  std::cerr << "mlx-scamp-native: computing on MLX Metal device 0\n";
  const auto compute_start = std::chrono::steady_clock::now();
  SCAMP::do_SCAMP(&arguments, {0}, 0);
  std::cerr << "mlx-scamp-native: computation finished in " << std::fixed
            << std::setprecision(3) << SecondsSince(compute_start) << "s\n";

  std::cerr << "mlx-scamp-native: staging outputs\n";
  std::vector<StagedOutput> staged;
  staged.reserve(options.keep_rows ? 4 : 2);
  try {
    StageProfile(options.output_a, options.output_a_index,
                 arguments.profile_a.data.at(0).uint64_value,
                 options.output_pearson, arguments.window, &staged);
    if (options.keep_rows) {
      StageProfile(options.output_b, options.output_b_index,
                   arguments.profile_b.data.at(0).uint64_value,
                   options.output_pearson, arguments.window, &staged);
    }
    // Computation and output staging can be long. Recheck identities before
    // the first destination rename so a path swapped to an input in that
    // interval is rejected without touching either file.
    ValidatePathSafety(options);
    CommitOutputs(&staged);
  } catch (...) {
    CleanStaged(&staged);
    throw;
  }
  std::cerr << "mlx-scamp-native: committed " << staged.size()
            << " output files in " << std::fixed << std::setprecision(3)
            << SecondsSince(start) << "s\n";
  return 0;
}

} // namespace mlx_scamp::cli
