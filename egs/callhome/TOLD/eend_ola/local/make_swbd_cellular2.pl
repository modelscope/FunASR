#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
#
# Copyright   2013   Daniel Povey
# Apache 2.0

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2004S07> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2004S07 data/swbd_cellular2_train\n";
  exit(1);
}
($db_base, $out_dir) = @ARGV;

if (system("mkdir -p $out_dir")) {
  die "Error making directory $out_dir";
}

open(CS, "<$db_base/docs/swb_callstats.tbl") || die  "Could not open $db_base/docs/swb_callstats.tbl";
open(GNDR, ">$out_dir/spk2gender") || die "Could not open the output file $out_dir/spk2gender";
open(SPKR, ">$out_dir/utt2spk") || die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">$out_dir/wav.scp") || die "Could not open the output file $out_dir/wav.scp";

@badAudio=("45024", "40022");

while (<CS>) {
  $line = $_ ;
  @A = split(",", $line);
  if (/$A[0]/i ~~ @badAudio) {
    # do nothing
  } else {
    $wav = "sw_" . $A[0];
    $spkr1= "sw_" . $A[1];
    $spkr2= "sw_" . $A[2];
    $gender1 = $A[3];
    $gender2 = $A[4];
    if ($A[3] eq "M") {
      $gender1 = "m";
    } elsif ($A[3] eq "F") {
      $gender1 = "f";
    } else {
      die "Unknown Gender in $line";
    }
    if ($A[4] eq "M") {
      $gender2 = "m";
    } elsif ($A[4] eq "F") {
      $gender2 = "f";
    } else {
      die "Unknown Gender in $line";
    }
    if (-e "$db_base/data/$wav.sph") {
      $uttId = $spkr1 . "-swbdc_" . $wav ."_1";
      if (!$spk2gender{$spkr1}) {
        $spk2gender{$spkr1} = $gender1;
        print GNDR "$spkr1"," $gender1\n";
      }
      print WAV "$uttId"," sph2pipe -f wav -p -c 1 $db_base/data/$wav.sph |\n";
      print SPKR "$uttId"," $spkr1","\n";

      $uttId = $spkr2 . "-swbdc_" . $wav ."_2";
      if (!$spk2gender{$spkr2}) {
        $spk2gender{$spkr2} = $gender2;
        print GNDR "$spkr2"," $gender2\n";
      }
      print WAV "$uttId"," sph2pipe -f wav -p -c 2 $db_base/data/$wav.sph |\n";
      print SPKR "$uttId"," $spkr2","\n";
    } else {
      print STDERR "Missing $db_base/data/$wav.sph\n";
    }
  }
}

close(WAV) || die;
close(SPKR) || die;
close(GNDR) || die;
if (system("utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
if (system("utils/fix_data_dir.sh $out_dir") != 0) {
  die "Error fixing data dir $out_dir";
}
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
