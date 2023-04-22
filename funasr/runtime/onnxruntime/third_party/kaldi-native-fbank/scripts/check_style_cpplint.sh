#!/bin/bash
#
# Copyright      2020  Mobvoi Inc. (authors: Fangjun Kuang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
#
# (1) To check files of the last commit
#  ./scripts/check_style_cpplint.sh
#
# (2) To check changed files not committed yet
#  ./scripts/check_style_cpplint.sh 1
#
# (3) To check all files in the project
#  ./scripts/check_style_cpplint.sh 2


cpplint_version="1.5.4"
cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
kaldi_native_fbank_dir=$(cd $cur_dir/.. && pwd)

build_dir=$kaldi_native_fbank_dir/build
mkdir -p $build_dir

cpplint_src=$build_dir/cpplint-${cpplint_version}/cpplint.py

if [ ! -d "$build_dir/cpplint-${cpplint_version}" ]; then
  pushd $build_dir
  if command -v wget &> /dev/null; then
    wget https://github.com/cpplint/cpplint/archive/${cpplint_version}.tar.gz
  elif command -v curl &> /dev/null; then
    curl -O -SL https://github.com/cpplint/cpplint/archive/${cpplint_version}.tar.gz
  else
    echo "Please install wget or curl to download cpplint"
    exit 1
  fi
  tar xf ${cpplint_version}.tar.gz
  rm ${cpplint_version}.tar.gz

  # cpplint will report the following error for: __host__ __device__ (
  #
  #     Extra space before ( in function call  [whitespace/parens] [4]
  #
  # the following patch disables the above error
  sed -i "3490i\        not Search(r'__host__ __device__\\\s+\\\(', fncall) and" $cpplint_src
  popd
fi

source $kaldi_native_fbank_dir/scripts/utils.sh

# return true if the given file is a c++ source file
# return false otherwise
function is_source_code_file() {
  case "$1" in
    *.cc|*.h|*.cu)
      echo true;;
    *)
      echo false;;
  esac
}

function check_style() {
  python3 $cpplint_src $1 || abort $1
}

function check_last_commit() {
  files=$(git diff HEAD^1 --name-only --diff-filter=ACDMRUXB)
  echo $files
}

function check_current_dir() {
  files=$(git status -s -uno --porcelain | awk '{
  if (NF == 4) {
    # a file has been renamed
    print $NF
  } else {
    print $2
  }}')

  echo $files
}

function do_check() {
  case "$1" in
    1)
      echo "Check changed files"
      files=$(check_current_dir)
      ;;
    2)
      echo "Check all files"
      files=$(find $kaldi_native_fbank_dir/kaldi-native-fbank -name "*.h" -o -name "*.cc" -o -name "*.cu")
      ;;
    *)
      echo "Check last commit"
      files=$(check_last_commit)
      ;;
  esac

  for f in $files; do
    need_check=$(is_source_code_file $f)
    if $need_check; then
      [[ -f $f ]] && check_style $f
    fi
  done
}

function main() {
  do_check $1

  ok "Great! Style check passed!"
}

cd $kaldi_native_fbank_dir

main $1
