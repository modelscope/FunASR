#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
#
# Copyright   2017   David Snyder
# Apache 2.0

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC98S75> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora3/LDC/LDC98S75 data/swbd2_phase1_train\n";
  exit(1);
}
($db_base, $out_dir) = @ARGV;

if (system("mkdir -p $out_dir")) {
  die "Error making directory $out_dir";
}

open(CS, "<$db_base/doc/callstat.tbl") || die  "Could not open $db_base/doc/callstat.tbl";
open(GNDR, ">$out_dir/spk2gender") || die "Could not open the output file $out_dir/spk2gender";
open(SPKR, ">$out_dir/utt2spk") || die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">$out_dir/wav.scp") || die "Could not open the output file $out_dir/wav.scp";

@badAudio = ("3", "4");

$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

if (system("find $db_base -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<$tmp_dir/sph.list") or die "cannot open wav list";

%wavs = ();
while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $uttId = $t1[0];
  $wavs{$uttId} = $sph;
}

while (<CS>) {
  $line = $_ ;
  @A = split(",", $line);
  @A1 = split("[./]",$A[0]);
  $wav = $A1[0];
  if (/$wav/i ~~ @badAudio) {
    # do nothing
    print "Bad Audio = $wav";
  } else {
    $spkr1= "sw_" . $A[2];
    $spkr2= "sw_" . $A[3];
    $gender1 = $A[5];
    $gender2 = $A[6];
    if ($gender1 eq "M") {
      $gender1 = "m";
    } elsif ($gender1 eq "F") {
      $gender1 = "f";
    } else {
      die "Unknown Gender in $line";
    }
    if ($gender2 eq "M") {
      $gender2 = "m";
    } elsif ($gender2 eq "F") {
      $gender2 = "f";
    } else {
      die "Unknown Gender in $line";
    }
    if (-e "$wavs{$wav}") {
      $uttId = $spkr1 ."_" . $wav ."_1";
      if (!$spk2gender{$spkr1}) {
        $spk2gender{$spkr1} = $gender1;
        print GNDR "$spkr1"," $gender1\n";
      }
      print WAV "$uttId"," sph2pipe -f wav -p -c 1 $wavs{$wav} |\n";
      print SPKR "$uttId"," $spkr1","\n";

      $uttId = $spkr2 . "_" . $wav ."_2";
      if (!$spk2gender{$spkr2}) {
        $spk2gender{$spkr2} = $gender2;
        print GNDR "$spkr2"," $gender2\n";
      }
      print WAV "$uttId"," sph2pipe -f wav -p -c 2 $wavs{$wav} |\n";
      print SPKR "$uttId"," $spkr2","\n";
    } else {
      print STDERR "Missing $wavs{$wav} for $wav\n";
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
