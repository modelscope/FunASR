# ------------------------------------------------------------------------------
# Add native rules to configure source files
def gflags_sources(namespace=["google", "gflags"]):
    native.genrule(
        name = "gflags_declare_h",
        srcs = ["src/gflags_declare.h.in"],
        outs = ["gflags_declare.h"],
        cmd  = ("awk '{ " +
                "gsub(/@GFLAGS_NAMESPACE@/, \"" + namespace[0] + "\"); " +
                "gsub(/@(HAVE_STDINT_H|HAVE_SYS_TYPES_H|HAVE_INTTYPES_H|GFLAGS_INTTYPES_FORMAT_C99)@/, \"1\"); " +
                "gsub(/@([A-Z0-9_]+)@/, \"0\"); " +
                "print; }' $(<) > $(@)")
    )
    gflags_ns_h_files = []
    for ns in namespace[1:]:
        gflags_ns_h_file = "gflags_{}.h".format(ns)
        native.genrule(
            name = gflags_ns_h_file.replace('.', '_'),
            srcs = ["src/gflags_ns.h.in"],
            outs = [gflags_ns_h_file],
            cmd  = ("awk '{ " +
                    "gsub(/@ns@/, \"" + ns + "\"); " +
                    "gsub(/@NS@/, \"" + ns.upper() + "\"); " +
                    "print; }' $(<) > $(@)")
        )
        gflags_ns_h_files.append(gflags_ns_h_file)
    native.genrule(
        name = "gflags_h",
        srcs = ["src/gflags.h.in"],
        outs = ["gflags.h"],
        cmd  = ("awk '{ " +
                "gsub(/@GFLAGS_ATTRIBUTE_UNUSED@/, \"\"); " +
                "gsub(/@INCLUDE_GFLAGS_NS_H@/, \"" + '\n'.join(["#include \\\"gflags/{}\\\"".format(hdr) for hdr in gflags_ns_h_files]) + "\"); " +
                "print; }' $(<) > $(@)")
    )
    native.genrule(
        name = "gflags_completions_h",
        srcs = ["src/gflags_completions.h.in"],
        outs = ["gflags_completions.h"],
        cmd  = "awk '{ gsub(/@GFLAGS_NAMESPACE@/, \"" + namespace[0] + "\"); print; }' $(<) > $(@)"
    )
    hdrs = [":gflags_h", ":gflags_declare_h", ":gflags_completions_h"]
    hdrs.extend([':' + hdr.replace('.', '_') for hdr in gflags_ns_h_files])
    srcs = [
        "src/config.h",
        "src/gflags.cc",
        "src/gflags_completions.cc",
        "src/gflags_reporting.cc",
        "src/mutex.h",
        "src/util.h",
    ] + select({
        "//:x64_windows": [
            "src/windows_port.cc",
            "src/windows_port.h",
        ],
        "//conditions:default": [],
    })
    return [hdrs, srcs]

# ------------------------------------------------------------------------------
# Add native rule to build gflags library
def gflags_library(hdrs=[], srcs=[], threads=1):
    name = "gflags"
    copts = [
        "-DGFLAGS_BAZEL_BUILD",
        "-DGFLAGS_INTTYPES_FORMAT_C99",
        "-DGFLAGS_IS_A_DLL=0",
        # macros otherwise defined by CMake configured defines.h file
        "-DHAVE_STDINT_H",
        "-DHAVE_SYS_TYPES_H",
        "-DHAVE_INTTYPES_H",
        "-DHAVE_SYS_STAT_H",
        "-DHAVE_STRTOLL",
        "-DHAVE_STRTOQ",
        "-DHAVE_RWLOCK",
    ] + select({
        "//:x64_windows": [
            "-DOS_WINDOWS",
        ],
        "//conditions:default": [
            "-DHAVE_UNISTD_H",
            "-DHAVE_FNMATCH_H",
            "-DHAVE_PTHREAD",
        ],
    })
    linkopts = []
    if threads:
        linkopts += select({
            "//:x64_windows": [],
            "//conditions:default": ["-lpthread"],
        })
    else:
        name += "_nothreads"
        copts += ["-DNO_THREADS"]
    native.cc_library(
        name       = name,
        hdrs       = hdrs,
        srcs       = srcs,
        copts      = copts,
        linkopts   = linkopts,
        visibility = ["//visibility:public"],
        include_prefix = 'gflags'
    )
