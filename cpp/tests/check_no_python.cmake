if(NOT DEFINED CLI OR NOT EXISTS "${CLI}")
  message(FATAL_ERROR "CLI executable was not provided")
endif()

execute_process(
  COMMAND otool -L "${CLI}"
  RESULT_VARIABLE OTOOL_RESULT
  OUTPUT_VARIABLE OTOOL_OUTPUT
  ERROR_VARIABLE OTOOL_ERROR)
if(NOT OTOOL_RESULT EQUAL 0)
  message(FATAL_ERROR "otool failed: ${OTOOL_ERROR}")
endif()
string(TOLOWER "${OTOOL_OUTPUT}" OTOOL_LOWER)
if(OTOOL_LOWER MATCHES "libpython|python[.]framework")
  message(FATAL_ERROR "native CLI links a Python runtime:\n${OTOOL_OUTPUT}")
endif()
message(STATUS "native CLI has no Python runtime linkage")
