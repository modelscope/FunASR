#!/bin/bash

# This is a port port maintainer's file, and is not part of OpenFST.
# Licensed under the same Apache 2.0 license (see the file COPYING),
# or just delete it if in legal doubt; you do not need it anyway.
# Copyright 2019 SmartAction LLC.

set -euo pipefail; shopt -s extglob

LF=$'\n'

say() { echo >&2 "$0: $@"; }
die() { echo >&2 "$0: error: $@"; exit 1; }

# This is Spar^H^H Windooows!
miss=(); for tool in gawk git grep gzip tar wget; do
  command -v $tool >/dev/null || miss+=($tool)
done
test ${#miss[@]} -eq 0 || die "missing required program(s): ${miss[@]}"
unset miss

[[ "$#" == [12] ]] || { 
  echo >&2 -e "Usage: $0 <version> [<revision>=1]${LF} e.g.: $0 1.7.0 1"
  exit 2; }

ver=$1
rev=${2:-1}
tag="orig/$ver.$rev"
url="http://www.openfst.org/twiki/bin/viewfile/FST/FstDownload?filename=openfst-${ver}.tar.gz;rev=${rev}"

grep -qP '^1(\.1?\d){3}$' <<<"$ver.$rev" ||
  die "revision $ver.$rev does not look correct to me"

pfx="$(git rev-parse --show-prefix)" && test -z "$pfx" ||
  die "must be in the root of the work tree"

# Must be on the 'original' branch, in sync with the main repo,
# and with no local changes, although we can ignore deletions,
# since we are deleting files before import anyway. We must be
# squeaky clean, lest stray random files be added. Also, ignore
# changes to this file to allow for debugging.
git fetch origin
git status --porcelain=2 --branch | gawk -v me=$0 >&2 '
  $0 == "# branch.head original" { branch_ok = 1 }
  $1$2 == "#branch.ab"           { ab_bad = $4 != "-0" }
  $1 == "#" { next }
  $1$2 == "1.D" { next }  # Deleted files are fine.
  $1$9 == "1_import_release.sh" { next }
  { modf_bad=1; nextfile }
  END {
    me = me ": error: "
    if (!branch_ok) print me "current branch must be \"original\""
    if (ab_bad) print me "current branch is not in sync with remote"
    if (modf_bad) print me "there are locally modified files"
    exit !(branch_ok && !ab_bad && !modf_bad) }'

git rev-parse -q --verify "refs/tags/$tag" >/dev/null &&
  die "tag $tag already exists"

say "Removing (almost) all files from worktree"
rm -rf -- !(.|..|.git|.gitignore|_import_release.sh)

say "Fetching and unpacking OpenFST v$ver r$rev${LF} ... from: $url"
wget -nv -O - -- "$url" | tar -xzf - --strip-components=1

test -f NEWS ||
  die "file NEWS was not unpacked. Something went horribly wrong."

message="Import v$ver r$rev${LF}${LF}$url"

say "committing and tagging new version"
git add -A
git commit -m "$message"
git tag -a "$tag" -m "$message"

say "summary of changes to the previous version, and a NEWS snippet"

git --no-pager show --stat
echo
git --no-pager show --format='' --unified=1 -w -- NEWS
echo

say "tag ${tag} created for the imported version. Import complete."
