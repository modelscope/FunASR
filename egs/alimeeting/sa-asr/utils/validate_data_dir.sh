#!/usr/bin/env bash

cmd="$@"

no_feats=false
no_wav=false
no_text=false
no_spk_sort=false
non_print=false


function show_help
{
      echo "Usage: $0 [--no-feats] [--no-text] [--non-print] [--no-wav] [--no-spk-sort] <data-dir>"
      echo "The --no-xxx options mean that the script does not require "
      echo "xxx.scp to be present, but it will check it if it is present."
      echo "--no-spk-sort means that the script does not require the utt2spk to be "
      echo "sorted by the speaker-id in addition to being sorted by utterance-id."
      echo "--non-print ignore the presence of non-printable characters."
      echo "By default, utt2spk is expected to be sorted by both, which can be "
      echo "achieved by making the speaker-id prefixes of the utterance-ids"
      echo "e.g.: $0 data/train"
}

while [ $# -ne 0 ] ; do
  case "$1" in
    "--no-feats")
      no_feats=true;
      ;;
    "--no-text")
      no_text=true;
      ;;
    "--non-print")
      non_print=true;
      ;;
    "--no-wav")
      no_wav=true;
      ;;
    "--no-spk-sort")
      no_spk_sort=true;
      ;;
    *)
      if ! [ -z "$data" ] ; then
        show_help;
        exit 1
      fi
      data=$1
      ;;
  esac
  shift
done



if [ ! -d $data ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -f $data/images.scp ]; then
  cmd=${cmd/--no-wav/}  # remove --no-wav if supplied
  image/validate_data_dir.sh $cmd
  exit $?
fi

for f in spk2utt utt2spk; do
  if [ ! -f $data/$f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
  if [ ! -s $data/$f ]; then
    echo "$0: empty file $f"
    exit 1;
  fi
done

! cat $data/utt2spk | awk '{if (NF != 2) exit(1); }' && \
  echo "$0: $data/utt2spk has wrong format." && exit;

ns=$(wc -l < $data/spk2utt)
if [ "$ns" == 1 ]; then
  echo "$0: WARNING: you have only one speaker.  This probably a bad idea."
  echo "   Search for the word 'bold' in http://kaldi-asr.org/doc/data_prep.html"
  echo "   for more information."
fi


tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT HUP INT PIPE TERM

export LC_ALL=C

function check_sorted_and_uniq {
  ! perl -ne '((substr $_,-1) eq "\n") or die "file $ARGV has invalid newline";' $1 && exit 1;
  ! awk '{print $1}' < $1 | sort -uC && echo "$0: file $1 is not sorted or has duplicates" && exit 1;
}

function partial_diff {
  diff -U1 $1 $2 | (head -n 6; echo "..."; tail -n 6)
  n1=`cat $1 | wc -l`
  n2=`cat $2 | wc -l`
  echo "[Lengths are $1=$n1 versus $2=$n2]"
}

check_sorted_and_uniq $data/utt2spk

if ! $no_spk_sort; then
  ! sort -k2 -C $data/utt2spk && \
     echo "$0: utt2spk is not in sorted order when sorted first on speaker-id " && \
     echo "(fix this by making speaker-ids prefixes of utt-ids)" && exit 1;
fi

check_sorted_and_uniq $data/spk2utt

! cmp -s <(cat $data/utt2spk | awk '{print $1, $2;}') \
     <(utils/spk2utt_to_utt2spk.pl $data/spk2utt)  && \
   echo "$0: spk2utt and utt2spk do not seem to match" && exit 1;

cat $data/utt2spk | awk '{print $1;}' > $tmpdir/utts

if [ ! -f $data/text ] && ! $no_text; then
  echo "$0: no such file $data/text (if this is by design, specify --no-text)"
  exit 1;
fi

num_utts=`cat $tmpdir/utts | wc -l`
if ! $no_text; then
  if ! $non_print; then
    if locale -a | grep "C.UTF-8" >/dev/null; then
      L=C.UTF-8
    else
      L=en_US.UTF-8
    fi
    n_non_print=$(LC_ALL="$L" grep -c '[^[:print:][:space:]]' $data/text) && \
    echo "$0: text contains $n_non_print lines with non-printable characters" &&\
    exit 1;
  fi
  utils/validate_text.pl $data/text || exit 1;
  check_sorted_and_uniq $data/text
  text_len=`cat $data/text | wc -l`
  illegal_sym_list="<s> </s> #0"
  for x in $illegal_sym_list; do
    if grep -w "$x" $data/text > /dev/null; then
      echo "$0: Error: in $data, text contains illegal symbol $x"
      exit 1;
    fi
  done
  awk '{print $1}' < $data/text > $tmpdir/utts.txt
  if ! cmp -s $tmpdir/utts{,.txt}; then
    echo "$0: Error: in $data, utterance lists extracted from utt2spk and text"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/utts{,.txt}
    exit 1;
  fi
fi

if [ -f $data/segments ] && [ ! -f $data/wav.scp ]; then
  echo "$0: in directory $data, segments file exists but no wav.scp"
  exit 1;
fi


if [ ! -f $data/wav.scp ] && ! $no_wav; then
  echo "$0: no such file $data/wav.scp (if this is by design, specify --no-wav)"
  exit 1;
fi

if [ -f $data/wav.scp ]; then
  check_sorted_and_uniq $data/wav.scp

  if grep -E -q '^\S+\s+~' $data/wav.scp; then
    # note: it's not a good idea to have any kind of tilde in wav.scp, even if
    # part of a command, as it would cause compatibility problems if run by
    # other users, but this used to be not checked for so we let it slide unless
    # it's something of the form "foo ~/foo.wav" (i.e. a plain file name) which
    # would definitely cause problems as the fopen system call does not do
    # tilde expansion.
    echo "$0: Please do not use tilde (~) in your wav.scp."
    exit 1;
  fi

  if [ -f $data/segments ]; then

    check_sorted_and_uniq $data/segments
    # We have a segments file -> interpret wav file as "recording-ids" not utterance-ids.
    ! cat $data/segments | \
      awk '{if (NF != 4 || $4 <= $3) { print "Bad line in segments file", $0; exit(1); }}' && \
      echo "$0: badly formatted segments file" && exit 1;

    segments_len=`cat $data/segments | wc -l`
    if [ -f $data/text ]; then
      ! cmp -s $tmpdir/utts <(awk '{print $1}' <$data/segments) && \
        echo "$0: Utterance list differs between $data/utt2spk and $data/segments " && \
        echo "$0: Lengths are $segments_len vs $num_utts" && \
        exit 1
    fi

    cat $data/segments | awk '{print $2}' | sort | uniq > $tmpdir/recordings
    awk '{print $1}' $data/wav.scp > $tmpdir/recordings.wav
    if ! cmp -s $tmpdir/recordings{,.wav}; then
      echo "$0: Error: in $data, recording-ids extracted from segments and wav.scp"
      echo "$0: differ, partial diff is:"
      partial_diff $tmpdir/recordings{,.wav}
      exit 1;
    fi
    if [ -f $data/reco2file_and_channel ]; then
      # this file is needed only for ctm scoring; it's indexed by recording-id.
      check_sorted_and_uniq $data/reco2file_and_channel
      ! cat $data/reco2file_and_channel | \
        awk '{if (NF != 3 || ($3 != "A" && $3 != "B" )) {
                if ( NF == 3 && $3 == "1" ) {
                  warning_issued = 1;
                } else {
                  print "Bad line ", $0; exit 1;
                }
              }
            }
            END {
              if (warning_issued == 1) {
                print "The channel should be marked as A or B, not 1! You should change it ASAP! "
              }
            }' && echo "$0: badly formatted reco2file_and_channel file" && exit 1;
      cat $data/reco2file_and_channel | awk '{print $1}' > $tmpdir/recordings.r2fc
      if ! cmp -s $tmpdir/recordings{,.r2fc}; then
        echo "$0: Error: in $data, recording-ids extracted from segments and reco2file_and_channel"
        echo "$0: differ, partial diff is:"
        partial_diff $tmpdir/recordings{,.r2fc}
        exit 1;
      fi
    fi
  else
    # No segments file -> assume wav.scp indexed by utterance.
    cat $data/wav.scp | awk '{print $1}' > $tmpdir/utts.wav
    if ! cmp -s $tmpdir/utts{,.wav}; then
      echo "$0: Error: in $data, utterance lists extracted from utt2spk and wav.scp"
      echo "$0: differ, partial diff is:"
      partial_diff $tmpdir/utts{,.wav}
      exit 1;
    fi

    if [ -f $data/reco2file_and_channel ]; then
      # this file is needed only for ctm scoring; it's indexed by recording-id.
      check_sorted_and_uniq $data/reco2file_and_channel
      ! cat $data/reco2file_and_channel | \
        awk '{if (NF != 3 || ($3 != "A" && $3 != "B" )) {
                if ( NF == 3 && $3 == "1" ) {
                  warning_issued = 1;
                } else {
                  print "Bad line ", $0; exit 1;
                }
              }
            }
            END {
              if (warning_issued == 1) {
                print "The channel should be marked as A or B, not 1! You should change it ASAP! "
              }
            }' && echo "$0: badly formatted reco2file_and_channel file" && exit 1;
      cat $data/reco2file_and_channel | awk '{print $1}' > $tmpdir/utts.r2fc
      if ! cmp -s $tmpdir/utts{,.r2fc}; then
        echo "$0: Error: in $data, utterance-ids extracted from segments and reco2file_and_channel"
        echo "$0: differ, partial diff is:"
        partial_diff $tmpdir/utts{,.r2fc}
        exit 1;
      fi
    fi
  fi
fi

if [ ! -f $data/feats.scp ] && ! $no_feats; then
  echo "$0: no such file $data/feats.scp (if this is by design, specify --no-feats)"
  exit 1;
fi

if [ -f $data/feats.scp ]; then
  check_sorted_and_uniq $data/feats.scp
  cat $data/feats.scp | awk '{print $1}' > $tmpdir/utts.feats
  if ! cmp -s $tmpdir/utts{,.feats}; then
    echo "$0: Error: in $data, utterance-ids extracted from utt2spk and features"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/utts{,.feats}
    exit 1;
  fi
fi


if [ -f $data/cmvn.scp ]; then
  check_sorted_and_uniq $data/cmvn.scp
  cat $data/cmvn.scp | awk '{print $1}' > $tmpdir/speakers.cmvn
  cat $data/spk2utt | awk '{print $1}' > $tmpdir/speakers
  if ! cmp -s $tmpdir/speakers{,.cmvn}; then
    echo "$0: Error: in $data, speaker lists extracted from spk2utt and cmvn"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/speakers{,.cmvn}
    exit 1;
  fi
fi

if [ -f $data/spk2gender ]; then
  check_sorted_and_uniq $data/spk2gender
  ! cat $data/spk2gender | awk '{if (!((NF == 2 && ($2 == "m" || $2 == "f")))) exit 1; }' && \
     echo "$0: Mal-formed spk2gender file" && exit 1;
  cat $data/spk2gender | awk '{print $1}' > $tmpdir/speakers.spk2gender
  cat $data/spk2utt | awk '{print $1}' > $tmpdir/speakers
  if ! cmp -s $tmpdir/speakers{,.spk2gender}; then
    echo "$0: Error: in $data, speaker lists extracted from spk2utt and spk2gender"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/speakers{,.spk2gender}
    exit 1;
  fi
fi

if [ -f $data/spk2warp ]; then
  check_sorted_and_uniq $data/spk2warp
  ! cat $data/spk2warp | awk '{if (!((NF == 2 && ($2 > 0.5 && $2 < 1.5)))){ print; exit 1; }}' && \
     echo "$0: Mal-formed spk2warp file" && exit 1;
  cat $data/spk2warp | awk '{print $1}' > $tmpdir/speakers.spk2warp
  cat $data/spk2utt | awk '{print $1}' > $tmpdir/speakers
  if ! cmp -s $tmpdir/speakers{,.spk2warp}; then
    echo "$0: Error: in $data, speaker lists extracted from spk2utt and spk2warp"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/speakers{,.spk2warp}
    exit 1;
  fi
fi

if [ -f $data/utt2warp ]; then
  check_sorted_and_uniq $data/utt2warp
  ! cat $data/utt2warp | awk '{if (!((NF == 2 && ($2 > 0.5 && $2 < 1.5)))){ print; exit 1; }}' && \
     echo "$0: Mal-formed utt2warp file" && exit 1;
  cat $data/utt2warp | awk '{print $1}' > $tmpdir/utts.utt2warp
  cat $data/utt2spk | awk '{print $1}' > $tmpdir/utts
  if ! cmp -s $tmpdir/utts{,.utt2warp}; then
    echo "$0: Error: in $data, utterance lists extracted from utt2spk and utt2warp"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/utts{,.utt2warp}
    exit 1;
  fi
fi

# check some optionally-required things
for f in vad.scp utt2lang utt2uniq; do
  if [ -f $data/$f ]; then
    check_sorted_and_uniq $data/$f
    if ! cmp -s <( awk '{print $1}' $data/utt2spk ) \
      <( awk '{print $1}' $data/$f ); then
      echo "$0: error: in $data, $f and utt2spk do not have identical utterance-id list"
      exit 1;
    fi
  fi
done


if [ -f $data/utt2dur ]; then
  check_sorted_and_uniq $data/utt2dur
  cat $data/utt2dur | awk '{print $1}' > $tmpdir/utts.utt2dur
  if ! cmp -s $tmpdir/utts{,.utt2dur}; then
    echo "$0: Error: in $data, utterance-ids extracted from utt2spk and utt2dur file"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/utts{,.utt2dur}
    exit 1;
  fi
  cat $data/utt2dur | \
    awk '{ if (NF != 2 || !($2 > 0)) { print "Bad line utt2dur:" NR ":" $0; exit(1) }}' || exit 1
fi

if [ -f $data/utt2num_frames ]; then
  check_sorted_and_uniq $data/utt2num_frames
  cat $data/utt2num_frames | awk '{print $1}' > $tmpdir/utts.utt2num_frames
  if ! cmp -s $tmpdir/utts{,.utt2num_frames}; then
    echo "$0: Error: in $data, utterance-ids extracted from utt2spk and utt2num_frames file"
    echo "$0: differ, partial diff is:"
    partial_diff $tmpdir/utts{,.utt2num_frames}
    exit 1
  fi
  awk <$data/utt2num_frames '{
    if (NF != 2 || !($2 > 0) || $2 != int($2)) {
      print "Bad line utt2num_frames:" NR ":" $0
      exit 1 } }' || exit 1
fi

if [ -f $data/reco2dur ]; then
  check_sorted_and_uniq $data/reco2dur
  cat $data/reco2dur | awk '{print $1}' > $tmpdir/recordings.reco2dur
  if [ -f $tmpdir/recordings ]; then
    if ! cmp -s $tmpdir/recordings{,.reco2dur}; then
      echo "$0: Error: in $data, recording-ids extracted from segments and reco2dur file"
      echo "$0: differ, partial diff is:"
      partial_diff $tmpdir/recordings{,.reco2dur}
    exit 1;
    fi
  else
    if ! cmp -s $tmpdir/{utts,recordings.reco2dur}; then
      echo "$0: Error: in $data, recording-ids extracted from wav.scp and reco2dur file"
      echo "$0: differ, partial diff is:"
      partial_diff $tmpdir/{utts,recordings.reco2dur}
    exit 1;
    fi
  fi
  cat $data/reco2dur | \
    awk '{ if (NF != 2 || !($2 > 0)) { print "Bad line : " $0; exit(1) }}' || exit 1
fi


echo "$0: Successfully validated data-directory $data"
