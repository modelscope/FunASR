#!/usr/bin/env bash

# Copyright 2016  Johns Hopkins University (author: Daniel Povey)
#           2018  Andrea Carmantini
# Apache 2.0

# This script operates on a data directory, such as in data/train/, and adds the
# reco2dur file if it does not already exist.  The file 'reco2dur' maps from
# recording to the duration of the recording in seconds.  This script works it
# out from the 'wav.scp' file, or, if utterance-ids are the same as recording-ids, from the
# utt2dur file (it first tries interrogating the headers, and if this fails, it reads the wave
# files in entirely.)
# We could use durations from segments file, but that's not the duration of the recordings
# but the sum of utterance lenghts (silence in between could be excluded from segments)
# For sum of utterance lenghts:
# awk 'FNR==NR{uttdur[$1]=$2;next}
# { for(i=2;i<=NF;i++){dur+=uttdur[$i];}
#   print $1 FS dur; dur=0  }'  $data/utt2dur $data/reco2utt


frame_shift=0.01
cmd=run.pl
nj=4

. utils/parse_options.sh
. ./path.sh

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <datadir>"
  echo "e.g.:"
  echo " $0 data/train"
  echo " Options:"
  echo " --frame-shift      # frame shift in seconds. Only relevant when we are"
  echo "                    # getting duration from feats.scp (default: 0.01). "
  exit 1
fi

export LC_ALL=C

data=$1


if [ -s $data/reco2dur ] && \
  [ $(wc -l < $data/wav.scp) -eq $(wc -l < $data/reco2dur) ]; then
  echo "$0: $data/reco2dur already exists with the expected length.  We won't recompute it."
  exit 0;
fi

if [ -s $data/utt2dur ] && \
   [ $(wc -l < $data/utt2spk) -eq $(wc -l < $data/utt2dur) ] && \
   [ ! -s $data/segments ]; then

  echo "$0: $data/wav.scp indexed by utt-id; copying utt2dur to reco2dur"
  cp $data/utt2dur $data/reco2dur && exit 0;

elif [ -f $data/wav.scp ]; then
  echo "$0: obtaining durations from recordings"

  # if the wav.scp contains only lines of the form
  # utt1  /foo/bar/sph2pipe -f wav /baz/foo.sph |
  if cat $data/wav.scp | perl -e '
     while (<>) { s/\|\s*$/ |/;  # make sure final | is preceded by space.
             @A = split; if (!($#A == 5 && $A[1] =~ m/sph2pipe$/ &&
                               $A[2] eq "-f" && $A[3] eq "wav" && $A[5] eq "|")) { exit(1); }
             $reco = $A[0]; $sphere_file = $A[4];

             if (!open(F, "<$sphere_file")) { die "Error opening sphere file $sphere_file"; }
             $sample_rate = -1;  $sample_count = -1;
             for ($n = 0; $n <= 30; $n++) {
                $line = <F>;
                if ($line =~ m/sample_rate -i (\d+)/) { $sample_rate = $1; }
                if ($line =~ m/sample_count -i (\d+)/) { $sample_count = $1; }
                if ($line =~ m/end_head/) { break; }
             }
             close(F);
             if ($sample_rate == -1 || $sample_count == -1) {
               die "could not parse sphere header from $sphere_file";
             }
             $duration = $sample_count * 1.0 / $sample_rate;
             print "$reco $duration\n";
     } ' > $data/reco2dur; then
    echo "$0: successfully obtained recording lengths from sphere-file headers"
  else
    echo "$0: could not get recording lengths from sphere-file headers, using wav-to-duration"
    if ! command -v wav-to-duration >/dev/null; then
      echo  "$0: wav-to-duration is not on your path"
      exit 1;
    fi

    read_entire_file=false
    if grep -q 'sox.*speed' $data/wav.scp; then
      read_entire_file=true
      echo "$0: reading from the entire wav file to fix the problem caused by sox commands with speed perturbation. It is going to be slow."
      echo "... It is much faster if you call get_reco2dur.sh *before* doing the speed perturbation via e.g. perturb_data_dir_speed.sh or "
      echo "... perturb_data_dir_speed_3way.sh."
    fi

    num_recos=$(wc -l <$data/wav.scp)
    if [ $nj -gt $num_recos ]; then
      nj=$num_recos
    fi

    temp_data_dir=$data/wav${nj}split
    wavscps=$(for n in `seq $nj`; do echo $temp_data_dir/$n/wav.scp; done)
    subdirs=$(for n in `seq $nj`; do echo $temp_data_dir/$n; done)

    if ! mkdir -p $subdirs >&/dev/null; then
	for n in `seq $nj`; do
	    mkdir -p $temp_data_dir/$n
	done
    fi

    utils/split_scp.pl $data/wav.scp $wavscps


    $cmd JOB=1:$nj $data/log/get_reco_durations.JOB.log \
      wav-to-duration --read-entire-file=$read_entire_file \
      scp:$temp_data_dir/JOB/wav.scp ark,t:$temp_data_dir/JOB/reco2dur || \
        { echo "$0: there was a problem getting the durations"; exit 1; } # This could

    for n in `seq $nj`; do
      cat $temp_data_dir/$n/reco2dur
    done > $data/reco2dur
  fi
  rm -r $temp_data_dir
else
  echo "$0: Expected $data/wav.scp to exist"
  exit 1
fi

len1=$(wc -l < $data/wav.scp)
len2=$(wc -l < $data/reco2dur)
if [ "$len1" != "$len2" ]; then
  echo "$0: warning: length of reco2dur does not equal that of wav.scp, $len2 != $len1"
  if [ $len1 -gt $[$len2*2] ]; then
    echo "$0: less than half of recordings got a duration: failing."
    exit 1
  fi
fi

echo "$0: computed $data/reco2dur"

exit 0
