#ifndef DLL_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define DLL_H_62B23520_7C8E_11DE_8A39_0800200C9A66

// Definition YAML_CPP_STATIC_DEFINE using to building YAML-CPP as static
// library (definition created by CMake or defined manually)

// Definition yaml_cpp_EXPORTS using to building YAML-CPP as dll/so library
// (definition created by CMake or defined manually)

#ifdef YAML_CPP_STATIC_DEFINE
#  define YAML_CPP_API
#  define YAML_CPP_NO_EXPORT
#else
#  if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#    ifndef YAML_CPP_API
#      ifdef yaml_cpp_EXPORTS
         /* We are building this library */
#        pragma message( "Defining YAML_CPP_API for DLL export" )
#        define YAML_CPP_API __declspec(dllexport)
#      else
         /* We are using this library */
#        pragma message( "Defining YAML_CPP_API for DLL import" )
#        define YAML_CPP_API __declspec(dllimport)
#      endif
#    endif
#    ifndef YAML_CPP_NO_EXPORT
#      define YAML_CPP_NO_EXPORT
#    endif
#  else /* No _MSC_VER */
#    ifndef YAML_CPP_API
#      ifdef yaml_cpp_EXPORTS
         /* We are building this library */
#        define YAML_CPP_API __attribute__((visibility("default")))
#      else
         /* We are using this library */
#        define YAML_CPP_API __attribute__((visibility("default")))
#      endif
#    endif
#    ifndef YAML_CPP_NO_EXPORT
#      define YAML_CPP_NO_EXPORT __attribute__((visibility("hidden")))
#    endif
#  endif /* _MSC_VER */
#endif   /* YAML_CPP_STATIC_DEFINE */

#ifndef YAML_CPP_DEPRECATED
#  ifdef _MSC_VER
#    define YAML_CPP_DEPRECATED __declspec(deprecated)
#  else
#    define YAML_CPP_DEPRECATED __attribute__ ((__deprecated__))
#  endif
#endif

#ifndef YAML_CPP_DEPRECATED_EXPORT
#  define YAML_CPP_DEPRECATED_EXPORT YAML_CPP_API YAML_CPP_DEPRECATED
#endif

#ifndef YAML_CPP_DEPRECATED_NO_EXPORT
#  define YAML_CPP_DEPRECATED_NO_EXPORT YAML_CPP_NO_EXPORT YAML_CPP_DEPRECATED
#endif

#endif /* DLL_H_62B23520_7C8E_11DE_8A39_0800200C9A66 */
