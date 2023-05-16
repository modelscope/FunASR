# - Try to find libunwind
# Once done this will define
#
#  Unwind_FOUND - system has libunwind
#  unwind::unwind - cmake target for libunwind

include (FindPackageHandleStandardArgs)

find_path (Unwind_INCLUDE_DIR NAMES unwind.h libunwind.h DOC "unwind include directory")
find_library (Unwind_LIBRARY NAMES unwind DOC "unwind library")

mark_as_advanced (Unwind_INCLUDE_DIR Unwind_LIBRARY)

# Extract version information
if (Unwind_LIBRARY)
  set (_Unwind_VERSION_HEADER ${Unwind_INCLUDE_DIR}/libunwind-common.h)

  if (EXISTS ${_Unwind_VERSION_HEADER})
    file (READ ${_Unwind_VERSION_HEADER} _Unwind_VERSION_CONTENTS)

    string (REGEX REPLACE ".*#define UNW_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1"
      Unwind_VERSION_MAJOR "${_Unwind_VERSION_CONTENTS}")
    string (REGEX REPLACE ".*#define UNW_VERSION_MINOR[ \t]+([0-9]+).*" "\\1"
      Unwind_VERSION_MINOR "${_Unwind_VERSION_CONTENTS}")
    string (REGEX REPLACE ".*#define UNW_VERSION_EXTRA[ \t]+([0-9]+).*" "\\1"
      Unwind_VERSION_PATCH "${_Unwind_VERSION_CONTENTS}")

    set (Unwind_VERSION ${Unwind_VERSION_MAJOR}.${Unwind_VERSION_MINOR})

    if (CMAKE_MATCH_0)
      # Third version component may be empty
      set (Unwind_VERSION ${Unwind_VERSION}.${Unwind_VERSION_PATCH})
      set (Unwind_VERSION_COMPONENTS 3)
    else (CMAKE_MATCH_0)
      set (Unwind_VERSION_COMPONENTS 2)
    endif (CMAKE_MATCH_0)
  endif (EXISTS ${_Unwind_VERSION_HEADER})
endif (Unwind_LIBRARY)

# handle the QUIETLY and REQUIRED arguments and set Unwind_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args (Unwind
  REQUIRED_VARS Unwind_INCLUDE_DIR Unwind_LIBRARY
  VERSION_VAR Unwind_VERSION
)

if (Unwind_FOUND)
  if (NOT TARGET unwind::unwind)
    add_library (unwind::unwind INTERFACE IMPORTED)

    set_property (TARGET unwind::unwind PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES ${Unwind_INCLUDE_DIR}
    )
    set_property (TARGET unwind::unwind PROPERTY
      INTERFACE_LINK_LIBRARIES ${Unwind_LIBRARY}
    )
    set_property (TARGET unwind::unwind PROPERTY
      IMPORTED_CONFIGURATIONS RELEASE
    )
  endif (NOT TARGET unwind::unwind)
endif (Unwind_FOUND)
