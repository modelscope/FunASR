#!/usr/bin/env perl

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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


if ($ARGV[0] eq "--srand") {
  $n = $ARGV[1];
  $n =~ m/\d+/ || die "Bad argument to --srand option: \"$n\"";
  srand($ARGV[1]);
  shift;
  shift;
} else {
  srand(0); # Gives inconsistent behavior if we don't seed.
}

if (@ARGV > 1 || $ARGV[0] =~ m/^-.+/) { # >1 args, or an option we
  # don't understand.
  print "Usage: shuffle_list.pl [--srand N] [input file]  > output\n";
  print "randomizes the order of lines of input.\n";
  exit(1);
}

@lines;
while (<>) {
  push @lines, [ (rand(), $_)] ;
}

@lines = sort { $a->[0] cmp $b->[0] } @lines;
foreach $l (@lines) {
    print $l->[1];
}