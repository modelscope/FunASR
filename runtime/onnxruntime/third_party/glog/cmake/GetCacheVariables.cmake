cmake_policy (PUSH)
cmake_policy (VERSION 3.16...3.27)

include (CMakeParseArguments)

function (get_cache_variables _CACHEVARS)
  set (_SINGLE)
  set (_MULTI EXCLUDE)
  set (_OPTIONS)

  cmake_parse_arguments (_ARGS "${_OPTIONS}" "${_SINGLE}" "${_MULTI}" ${ARGS} ${ARGN})

  get_cmake_property (_VARIABLES VARIABLES)

  set (CACHEVARS)

  foreach (_VAR ${_VARIABLES})
    if (DEFINED _ARGS_EXCLUDE)
      if ("${_VAR}" IN_LIST _ARGS_EXCLUDE)
        continue ()
      endif ("${_VAR}" IN_LIST _ARGS_EXCLUDE)
    endif (DEFINED _ARGS_EXCLUDE)

    get_property (_CACHEVARTYPE CACHE ${_VAR} PROPERTY TYPE)

    if ("${_CACHEVARTYPE}" STREQUAL INTERNAL OR
        "${_CACHEVARTYPE}" STREQUAL STATIC OR
        "${_CACHEVARTYPE}" STREQUAL UNINITIALIZED)
        continue ()
    endif ("${_CACHEVARTYPE}" STREQUAL INTERNAL OR
        "${_CACHEVARTYPE}" STREQUAL STATIC OR
        "${_CACHEVARTYPE}" STREQUAL UNINITIALIZED)

    get_property (_CACHEVARVAL CACHE ${_VAR} PROPERTY VALUE)

    if ("${_CACHEVARVAL}" STREQUAL "")
      continue ()
    endif ("${_CACHEVARVAL}" STREQUAL "")

    get_property (_CACHEVARDOC CACHE ${_VAR} PROPERTY HELPSTRING)

    # Escape " in values
    string (REPLACE "\"" "\\\"" _CACHEVARVAL "${_CACHEVARVAL}")
    # Escape " in help strings
    string (REPLACE "\"" "\\\"" _CACHEVARDOC "${_CACHEVARDOC}")
    # Escape ; in values
    string (REPLACE ";" "\\\;" _CACHEVARVAL "${_CACHEVARVAL}")
    # Escape ; in help strings
    string (REPLACE ";" "\\\;" _CACHEVARDOC "${_CACHEVARDOC}")
    # Escape backslashes in values except those that are followed by a
    # quote.
    string (REGEX REPLACE "\\\\([^\"])" "\\\\\\1" _CACHEVARVAL "${_CACHEVARVAL}")
    # Escape backslashes in values that are followed by a letter to avoid
    # invalid escape sequence errors.
    string (REGEX REPLACE "\\\\([a-zA-Z])" "\\\\\\\\1" _CACHEVARVAL "${_CACHEVARVAL}")
    string (REPLACE "\\\\" "\\\\\\\\" _CACHEVARDOC "${_CACHEVARDOC}")

    if (NOT "${_CACHEVARTYPE}" STREQUAL BOOL)
      set (_CACHEVARVAL "\"${_CACHEVARVAL}\"")
    endif (NOT "${_CACHEVARTYPE}" STREQUAL BOOL)

    if (NOT "${_CACHEVARTYPE}" STREQUAL "" AND NOT "${_CACHEVARVAL}" STREQUAL "")
      set (CACHEVARS "${CACHEVARS}set (${_VAR} ${_CACHEVARVAL} CACHE ${_CACHEVARTYPE} \"${_CACHEVARDOC}\")\n")
    endif (NOT "${_CACHEVARTYPE}" STREQUAL "" AND NOT "${_CACHEVARVAL}" STREQUAL "")
  endforeach (_VAR)

  set (${_CACHEVARS} ${CACHEVARS} PARENT_SCOPE)
endfunction (get_cache_variables)

cmake_policy (POP)
