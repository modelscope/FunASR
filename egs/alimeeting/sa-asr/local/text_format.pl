#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
# Copyright Chao Weng 

# normalizations for hkust trascript
# see the docs/trans-guidelines.pdf for details

while (<STDIN>) {
  @A = split(" ", $_);
  if (@A == 1) {
    next;
  }
  print $_
}
