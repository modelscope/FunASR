#!/usr/bin/perl -w
# Copyright 2010-2011  Microsoft Corporation
#                2013  Johns Hopkins University (author: Daniel Povey)

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


# makes lexicon FST, in text form, from lexicon (pronunciation probabilities optional).

$pron_probs = 0;

if ($ARGV[0] eq "--pron-probs") {
  $pron_probs = 1;
  shift @ARGV;
}

if (@ARGV != 1 && @ARGV != 3 && @ARGV != 4) {
  print STDERR
    "Usage: make_lexicon_fst.pl [--pron-probs] lexicon.txt [silprob silphone [sil_disambig_sym]] >lexiconfst.txt
Creates a lexicon FST that transduces phones to words, and may allow optional silence. 
Note: ordinarily, each line of lexicon.txt is: word phone1 phone2 ... phoneN; if the --pron-probs option is
used, each line is: word pronunciation-probability phone1 phone2 ... phoneN.  The probability 'prob' will
typically be between zero and one, and note that it's generally helpful to normalize so the largest one
for each word is 1.0, but this is your responsibility.  The silence disambiguation symbol, e.g. something
like #5, is used only when creating a lexicon with disambiguation symbols, e.g. L_disambig.fst, and was
introduced to fix a particular case of non-determinism of decoding graphs.\n";
  exit(1);
}

$lexfn = shift @ARGV;
if (@ARGV == 0) {
  $silprob = 0.0;
} elsif (@ARGV == 2) { 
  ($silprob,$silphone) = @ARGV;
} else {
  ($silprob,$silphone,$sildisambig) = @ARGV;
}
if ($silprob != 0.0) {
  $silprob < 1.0 || die "Sil prob cannot be >= 1.0";
  $silcost = -log($silprob);
  $nosilcost = -log(1.0 - $silprob);
}


open(L, "<$lexfn") || die "Error opening lexicon $lexfn";


if ( $silprob == 0.0 ) { # No optional silences: just have one (loop+final) state which is numbered zero.
  $loopstate = 0;
  $nextstate = 1;               # next unallocated state.
  while (<L>) {
    @A = split(" ", $_);
    @A == 0 && die "Empty lexicon line.";
    $w = shift @A;
    if (! $pron_probs) {
      $pron_cost = 0.0;
    } else {
      $pron_prob = shift @A;
      if (! defined $pron_prob || !($pron_prob > 0.0 && $pron_prob <= 1.0)) {
        die "Bad pronunciation probability in line $_";
      }
      $pron_cost = -log($pron_prob);
    }
    if ($pron_cost != 0.0) { $pron_cost_string = "\t$pron_cost"; } else { $pron_cost_string = ""; }
    
    $s = $loopstate;
    $word_or_eps = $w;
    while (@A > 0) {
      $p = shift @A;
      if (@A > 0) {
        $ns = $nextstate++;
      } else {
        $ns = $loopstate;
      }
      print "$s\t$ns\t$p\t$word_or_eps$pron_cost_string\n";
      $word_or_eps = "<eps>";
      $pron_cost_string = ""; # so we only print it on the first arc of the word.
      $s = $ns;
    }
  }
  print "$loopstate\t0\n";      # final-cost.
} else {                        # have silence probs.
  $startstate = 0;
  $loopstate = 1;
  $silstate = 2;   # state from where we go to loopstate after emitting silence.
  print "$startstate\t$loopstate\t<eps>\t<eps>\t$nosilcost\n"; # no silence.
  if (!defined $sildisambig) {
    print "$startstate\t$loopstate\t$silphone\t<eps>\t$silcost\n"; # silence.
    print "$silstate\t$loopstate\t$silphone\t<eps>\n";             # no cost.
    $nextstate = 3;
  } else {
    $disambigstate = 3;
    $nextstate = 4;
    print "$startstate\t$disambigstate\t$silphone\t<eps>\t$silcost\n"; # silence.
    print "$silstate\t$disambigstate\t$silphone\t<eps>\n"; # no cost.
    print "$disambigstate\t$loopstate\t$sildisambig\t<eps>\n"; # silence disambiguation symbol.
  }
  while (<L>) {
    @A = split(" ", $_);
    $w = shift @A;
    if (! $pron_probs) {
      $pron_cost = 0.0;
    } else {
      $pron_prob = shift @A;
      if (! defined $pron_prob || !($pron_prob > 0.0 && $pron_prob <= 1.0)) {
        die "Bad pronunciation probability in line $_";
      }
      $pron_cost = -log($pron_prob);
    }
    if ($pron_cost != 0.0) { $pron_cost_string = "\t$pron_cost"; } else { $pron_cost_string = ""; }
    $s = $loopstate;
    $word_or_eps = $w;
    while (@A > 0) {
      $p = shift @A;
      if (@A > 0) {
        $ns = $nextstate++;
        print "$s\t$ns\t$p\t$word_or_eps$pron_cost_string\n";
        $word_or_eps = "<eps>";
        $pron_cost_string = ""; $pron_cost = 0.0; # so we only print it the 1st time.
        $s = $ns;
      } else {
        if (!defined($silphone) || $p ne $silphone) {
          # This is non-deterministic but relatively compact,
          # and avoids epsilons.
          $local_nosilcost = $nosilcost + $pron_cost;
          $local_silcost = $silcost + $pron_cost;
          print "$s\t$loopstate\t$p\t$word_or_eps\t$local_nosilcost\n";
          print "$s\t$silstate\t$p\t$word_or_eps\t$local_silcost\n";
        } else {
          # no point putting opt-sil after silence word.
          print "$s\t$loopstate\t$p\t$word_or_eps$pron_cost_string\n";
        }
      }
    }
  }
  print "$loopstate\t0\n";      # final-cost.
}
