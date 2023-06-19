#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This program is a bit like ./sym2int.pl in that it applies a map
# to things in a file, but it's a bit more general in that it doesn't
# assume the things being mapped to are single tokens, they could
# be sequences of tokens.  See the usage message.


$permissive = 0;

for ($x = 0; $x <= 2; $x++) {

  if (@ARGV > 0 && $ARGV[0] eq "-f") {
    shift @ARGV;
    $field_spec = shift @ARGV;
    if ($field_spec =~ m/^\d+$/) {
      $field_begin = $field_spec - 1; $field_end = $field_spec - 1;
    }
    if ($field_spec =~ m/^(\d*)[-:](\d*)/) { # accept e.g. 1:10 as a courtesty (properly, 1-10)
      if ($1 ne "") {
        $field_begin = $1 - 1;  # Change to zero-based indexing.
      }
      if ($2 ne "") {
        $field_end = $2 - 1;    # Change to zero-based indexing.
      }
    }
    if (!defined $field_begin && !defined $field_end) {
      die "Bad argument to -f option: $field_spec";
    }
  }

  if (@ARGV > 0 && $ARGV[0] eq '--permissive') {
    shift @ARGV;
    # Mapping is optional (missing key is printed to output)
    $permissive = 1;
  }
}

if(@ARGV != 1) {
  print STDERR "Invalid usage: " . join(" ", @ARGV) . "\n";
  print STDERR <<'EOF';
Usage: apply_map.pl [options] map <input >output
 options: [-f <field-range> ] [--permissive]
   This applies a map to some specified fields of some input text:
   For each line in the map file: the first field is the thing we
   map from, and the remaining fields are the sequence we map it to.
   The -f (field-range) option says which fields of the input file the map
   map should apply to.
   If the --permissive option is supplied, fields which are not present
   in the map will be left as they were.
 Applies the map 'map' to all input text, where each line of the map
 is interpreted as a map from the first field to the list of the other fields
 Note: <field-range> can look like 4-5, or 4-, or 5-, or 1, it means the field
 range in the input to apply the map to.
 e.g.: echo A B | apply_map.pl a.txt
 where a.txt is:
 A a1 a2
 B b
 will produce:
 a1 a2 b
EOF
  exit(1);
}

($map_file) = @ARGV;
open(M, "<$map_file") || die "Error opening map file $map_file: $!";

while (<M>) {
  @A = split(" ", $_);
  @A >= 1 || die "apply_map.pl: empty line.";
  $i = shift @A;
  $o = join(" ", @A);
  $map{$i} = $o;
}

while(<STDIN>) {
  @A = split(" ", $_);
  for ($x = 0; $x < @A; $x++) {
    if ( (!defined $field_begin || $x >= $field_begin)
         && (!defined $field_end || $x <= $field_end)) {
      $a = $A[$x];
      if (!defined $map{$a}) {
        if (!$permissive) {
          die "apply_map.pl: undefined key $a in $map_file\n";
        } else {
          print STDERR "apply_map.pl: warning! missing key $a in $map_file\n";
        }
      } else {
        $A[$x] = $map{$a};
      }
    }
  }
  print join(" ", @A) . "\n";
}
