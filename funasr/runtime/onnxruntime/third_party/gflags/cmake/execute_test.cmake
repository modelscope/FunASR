# ----------------------------------------------------------------------------
# sanitize string stored in variable for use in regular expression.
macro (sanitize_for_regex STRVAR)
  string (REGEX REPLACE "([.+*?^$()])" "\\\\\\1" ${STRVAR} "${${STRVAR}}")
endmacro ()

# ----------------------------------------------------------------------------
# script arguments
if (NOT COMMAND)
  message (FATAL_ERROR "Test command not specified!")
endif ()
if (NOT DEFINED EXPECTED_RC)
  set (EXPECTED_RC 0)
endif ()
if (EXPECTED_OUTPUT)
  sanitize_for_regex(EXPECTED_OUTPUT)
endif ()
if (UNEXPECTED_OUTPUT)
  sanitize_for_regex(UNEXPECTED_OUTPUT)
endif ()

# ----------------------------------------------------------------------------
# set a few environment variables (useful for --tryfromenv)
set (ENV{FLAGS_undefok} "foo,bar")
set (ENV{FLAGS_weirdo}  "")
set (ENV{FLAGS_version} "true")
set (ENV{FLAGS_help}    "false")

# ----------------------------------------------------------------------------
# execute test command
execute_process(
  COMMAND ${COMMAND}
  RESULT_VARIABLE RC
  OUTPUT_VARIABLE OUTPUT
  ERROR_VARIABLE  OUTPUT
)

if (OUTPUT)
  message ("${OUTPUT}")
endif ()

# ----------------------------------------------------------------------------
# check test result
if (NOT RC EQUAL EXPECTED_RC)
  string (REPLACE ";" " " COMMAND "${COMMAND}")
  message (FATAL_ERROR "Command:\n\t${COMMAND}\nExit status is ${RC}, expected ${EXPECTED_RC}")
endif ()
if (EXPECTED_OUTPUT AND NOT OUTPUT MATCHES "${EXPECTED_OUTPUT}")
  message (FATAL_ERROR "Test output does not match expected output: ${EXPECTED_OUTPUT}")
endif ()
if (UNEXPECTED_OUTPUT AND OUTPUT MATCHES "${UNEXPECTED_OUTPUT}")
  message (FATAL_ERROR "Test output matches unexpected output: ${UNEXPECTED_OUTPUT}")
endif ()