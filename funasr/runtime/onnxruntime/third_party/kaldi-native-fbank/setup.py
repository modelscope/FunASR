#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corporation (author: Fangjun Kuang)

import re

import setuptools

from cmake.cmake_extension import BuildExtension, bdist_wheel, cmake_extension


def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme


def get_package_version():
    with open("CMakeLists.txt") as f:
        content = f.read()

    match = re.search(r"set\(KALDI_NATIVE_FBANK_VERSION (.*)\)", content)
    latest_version = match.group(1).strip('"')
    return latest_version


package_name = "kaldi-native-fbank"

with open("kaldi-native-fbank/python/kaldi_native_fbank/__init__.py", "a") as f:
    f.write(f"__version__ = '{get_package_version()}'\n")

setuptools.setup(
    name=package_name,
    version=get_package_version(),
    author="Fangjun Kuang",
    author_email="csukuangfj@gmail.com",
    package_dir={"kaldi_native_fbank": "kaldi-native-fbank/python/kaldi_native_fbank"},
    packages=["kaldi_native_fbank"],
    url="https://github.com/csukuangfj/kaldi-native-fbank",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=[cmake_extension("_kaldi_native_fbank")],
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": bdist_wheel},
    zip_safe=False,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6.0",
    license="Apache licensed, as found in the LICENSE file",
)

# remove the line __version__ from kaldi-native-fbank/python/kaldi_native_fbank/__init__.py
with open("kaldi-native-fbank/python/kaldi_native_fbank/__init__.py", "r") as f:
    lines = f.readlines()

with open("kaldi-native-fbank/python/kaldi_native_fbank/__init__.py", "w") as f:
    for line in lines:
        if "__version__" in line:
            # skip __version__ = "x.x.x"
            continue
        f.write(line)
