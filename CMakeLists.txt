cmake_minimum_required(VERSION 3.24)

project(MLXNativeSCAMP VERSION 0.1.0 LANGUAGES CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  message(FATAL_ERROR
    "The native MLX/Metal C++ library requires Apple Silicon macOS")
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(MLX_SCAMP_DEVELOPER_TARGETS_DEFAULT ON)
if(NOT PROJECT_IS_TOP_LEVEL)
  set(MLX_SCAMP_DEVELOPER_TARGETS_DEFAULT OFF)
endif()
option(MLX_SCAMP_BUILD_EXAMPLES "Build native C++ examples"
       ${MLX_SCAMP_DEVELOPER_TARGETS_DEFAULT})
option(MLX_SCAMP_BUILD_TESTS "Build native C++ tests"
       ${MLX_SCAMP_DEVELOPER_TARGETS_DEFAULT})
set(MLX_SCAMP_PYTHON_EXECUTABLE "" CACHE FILEPATH
    "Python interpreter used only to locate an installed MLX wheel")

# MLX wheels include the C++ headers, libmlx, and MLXConfig.cmake. Consumers
# may set MLX_DIR/MLX_ROOT directly; for an in-tree build we also discover the
# package belonging to the selected Python interpreter.
find_package(MLX 0.23.2 CONFIG QUIET)
if(NOT MLX_FOUND)
  set(MLX_SCAMP_PYTHON_CANDIDATES)
  if(MLX_SCAMP_PYTHON_EXECUTABLE)
    list(APPEND MLX_SCAMP_PYTHON_CANDIDATES
         "${MLX_SCAMP_PYTHON_EXECUTABLE}")
  endif()
  if(DEFINED ENV{VIRTUAL_ENV})
    list(APPEND MLX_SCAMP_PYTHON_CANDIDATES
         "$ENV{VIRTUAL_ENV}/bin/python")
  endif()
  list(APPEND MLX_SCAMP_PYTHON_CANDIDATES python3 python)
  foreach(MLX_SCAMP_PYTHON IN LISTS MLX_SCAMP_PYTHON_CANDIDATES)
    execute_process(
      COMMAND "${MLX_SCAMP_PYTHON}" -c
              "import pathlib, mlx.core; print(pathlib.Path(mlx.core.__file__).parent)"
      RESULT_VARIABLE MLX_SCAMP_DISCOVERY_RESULT
      OUTPUT_VARIABLE MLX_SCAMP_PYTHON_PACKAGE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(MLX_SCAMP_DISCOVERY_RESULT STREQUAL "0" AND
       MLX_SCAMP_PYTHON_PACKAGE)
      list(PREPEND CMAKE_PREFIX_PATH "${MLX_SCAMP_PYTHON_PACKAGE}")
      break()
    endif()
  endforeach()
  find_package(MLX 0.23.2 CONFIG REQUIRED)
endif()

add_library(mlx_native_scamp_cpp cpp/src/scamp.cpp)
add_library(MLXNativeSCAMP::scamp ALIAS mlx_native_scamp_cpp)
set_target_properties(mlx_native_scamp_cpp PROPERTIES
  EXPORT_NAME scamp
  OUTPUT_NAME mlx_scamp
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  CXX_EXTENSIONS OFF
  BUILD_RPATH "${MLX_LIBRARY_DIRS}")
target_compile_features(mlx_native_scamp_cpp PUBLIC cxx_std_17)
target_compile_options(mlx_native_scamp_cpp PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)
target_include_directories(mlx_native_scamp_cpp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(mlx_native_scamp_cpp PUBLIC mlx)

if(MLX_SCAMP_BUILD_EXAMPLES)
  add_executable(mlx_scamp_cpp_example cpp/examples/indexed_1nn.cpp)
  target_link_libraries(mlx_scamp_cpp_example PRIVATE MLXNativeSCAMP::scamp)
endif()

if(MLX_SCAMP_BUILD_TESTS)
  enable_testing()
  add_executable(mlx_scamp_cpp_test cpp/tests/test_indexed_1nn.cpp)
  target_link_libraries(mlx_scamp_cpp_test PRIVATE MLXNativeSCAMP::scamp)
  add_test(NAME mlx_scamp_cpp_indexed_1nn COMMAND mlx_scamp_cpp_test)

  add_executable(mlx_scamp_cpp_api_test cpp/tests/test_upstream_api.cpp)
  target_link_libraries(mlx_scamp_cpp_api_test PRIVATE MLXNativeSCAMP::scamp)
  add_test(NAME mlx_scamp_cpp_upstream_api COMMAND mlx_scamp_cpp_api_test)
endif()

install(TARGETS mlx_native_scamp_cpp
  EXPORT MLXNativeSCAMPTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY cpp/include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT MLXNativeSCAMPTargets
  FILE MLXNativeSCAMPTargets.cmake
  NAMESPACE MLXNativeSCAMP::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MLXNativeSCAMP")

configure_package_config_file(
  cmake/MLXNativeSCAMPConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/MLXNativeSCAMPConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MLXNativeSCAMP")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MLXNativeSCAMPConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/MLXNativeSCAMPConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/MLXNativeSCAMPConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MLXNativeSCAMP")
