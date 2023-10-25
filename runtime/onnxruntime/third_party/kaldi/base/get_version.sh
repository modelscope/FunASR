#!/usr/bin/env bash

# Copyright 2017 University of Southern California (Author: Dogan Can)

# See ../../COPYING for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Kaldi versioning is loosely based on the semantic versioning scheme. This
# script tries to work out the version string from the partial version number
# specified in src/.version along with the recent git history. By convention
# src/.version specifies the first two components (MAJOR.MINOR) of the version
# number. The third component (PATCH) is determined by counting how many
# commits there are that are newer than than the last commit modifiying
# src/.version. If there are uncommitted changes in the src/ directory, then
# the version string is extended with a suffix (~N) specifiying the number of
# files with uncommitted changes. The last component of the version string is
# the abbreviated hash of the HEAD commit. If git history is not available or
# if the file src/.short_version exists, then the version string defaults to
# the number specified in src/.version.

set -e

# Change working directory to the directory where this script is located.
cd "$(dirname ${BASH_SOURCE[0]})"

# Read the partial version number specified in the first line of src/.version.
version=$(head -1 ../.version)

if [ -e ../.short_version ]; then
  echo "$0: File src/.short_version exists."
  echo "$0: Stopping the construction of full version number from git history."
elif ! [[ $version =~ ^[0-9][0-9]*.[0-9][0-9]*$ ]]; then
  echo "$0: The version number \"$version\" specified in src/.version is not" \
       "in MAJOR.MINOR format."
  echo "$0: Stopping the construction of full version number from git history."
elif ! which git >&/dev/null; then
  echo "$0: Git is not installed."
  echo "$0: Using the version number \"$version\" specified in src/.version."
elif [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" != true ]; then
  echo "$0: Git history is not available."
  echo "$0: Using the version number \"$version\" specified in src/.version."
else
  # Figure out patch number.
  version_commit=$(git log -1 --pretty=oneline ../.version | awk '{print $1}')
  patch_number=$(git rev-list ${version_commit}..HEAD | wc -l | awk '{print $1}')
  version="$version.$patch_number"

  # Check for uncommitted changes in src/.
  uncommitted_changes=$(git diff-index HEAD -- .. | wc -l | awk '{print $1}')
  if [ $uncommitted_changes -gt 0 ]; then
    # Add suffix ~N if there are N files in src/ with uncommitted changes
    version="$version~$uncommitted_changes"
  fi

  # Figure out HEAD commit SHA-1.
  head_commit=$(git log -1 --pretty=oneline | awk '{print $1}')
  head_commit_short=$(git log -1 --oneline --abbrev=4 | awk '{print $1}')
  version="$version-${head_commit_short}"
fi

# Empty version number is not allowed.
if [ -z "$version" ]; then
  version="?"
fi

# Write version info to a temporary file.
temp=$(mktemp /tmp/temp.XXXXXX)
trap 'rm -f "$temp"' EXIT
echo "// This file was automatically created by ./get_version.sh." > $temp
echo "// It is only included by ./kaldi-error.cc." >> $temp
echo "#define KALDI_VERSION \"$version\"" >> $temp
if [ -n "$head_commit" ]; then
  echo "#define KALDI_GIT_HEAD \"$head_commit\"" >> $temp
fi

# Overwrite ./version.h with the temporary file if they are different.
if ! cmp -s $temp version.h; then
  cp $temp version.h
  chmod 644 version.h
fi
exit 0
