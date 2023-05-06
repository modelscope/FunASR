#!/usr/bin/env bash

# This script makes sure that only the segments present in
# all of "feats.scp", "wav.scp" [if present], segments [if present]
# text, and utt2spk are present in any of them.
# It puts the original contents of data-dir into
# data-dir/.backup

cmd="$@"

utt_extra_files=
spk_extra_files=

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: utils/data/fix_data_dir.sh <data-dir>"
  echo "e.g.: utils/data/fix_data_dir.sh data/train"
  echo "This script helps ensure that the various files in a data directory"
  echo "are correctly sorted and filtered, for example removing utterances"
  echo "that have no features (if feats.scp is present)"
  exit 1
fi

data=$1

if [ -f $data/images.scp ]; then
  image/fix_data_dir.sh $cmd
  exit $?
fi

mkdir -p $data/.backup

[ ! -d $data ] && echo "$0: no such directory $data" && exit 1;

[ ! -f $data/utt2spk ] && echo "$0: no such file $data/utt2spk" && exit 1;

set -e -o pipefail -u

tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT HUP INT PIPE TERM

export LC_ALL=C

function check_sorted {
  file=$1
  sort -k1,1 -u <$file >$file.tmp
  if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
  else
    rm $file.tmp
  fi
}

for x in utt2spk spk2utt feats.scp text segments wav.scp cmvn.scp vad.scp \
    reco2file_and_channel spk2gender utt2lang utt2uniq utt2dur reco2dur utt2num_frames; do
  if [ -f $data/$x ]; then
    cp $data/$x $data/.backup/$x
    check_sorted $data/$x
  fi
done


function filter_file {
  filter=$1
  file_to_filter=$2
  cp $file_to_filter ${file_to_filter}.tmp
  utils/filter_scp.pl $filter ${file_to_filter}.tmp > $file_to_filter
  if ! cmp ${file_to_filter}.tmp  $file_to_filter >&/dev/null; then
    length1=$(cat ${file_to_filter}.tmp | wc -l)
    length2=$(cat ${file_to_filter} | wc -l)
    if [ $length1 -ne $length2 ]; then
      echo "$0: filtered $file_to_filter from $length1 to $length2 lines based on filter $filter."
    fi
  fi
  rm $file_to_filter.tmp
}

function filter_recordings {
  # We call this once before the stage when we filter on utterance-id, and once
  # after.

  if [ -f $data/segments ]; then
  # We have a segments file -> we need to filter this and the file wav.scp, and
  # reco2file_and_utt, if it exists, to make sure they have the same list of
  # recording-ids.

    if [ ! -f $data/wav.scp ]; then
      echo "$0: $data/segments exists but not $data/wav.scp"
      exit 1;
    fi
    awk '{print $2}' < $data/segments | sort | uniq > $tmpdir/recordings
    n1=$(cat $tmpdir/recordings | wc -l)
    [ ! -s $tmpdir/recordings ] && \
      echo "Empty list of recordings (bad file $data/segments)?" && exit 1;
    utils/filter_scp.pl $data/wav.scp $tmpdir/recordings > $tmpdir/recordings.tmp
    mv $tmpdir/recordings.tmp $tmpdir/recordings


    cp $data/segments{,.tmp}; awk '{print $2, $1, $3, $4}' <$data/segments.tmp >$data/segments
    filter_file $tmpdir/recordings $data/segments
    cp $data/segments{,.tmp}; awk '{print $2, $1, $3, $4}' <$data/segments.tmp >$data/segments
    rm $data/segments.tmp

    filter_file $tmpdir/recordings $data/wav.scp
    [ -f $data/reco2file_and_channel ] && filter_file $tmpdir/recordings $data/reco2file_and_channel
    [ -f $data/reco2dur ] && filter_file $tmpdir/recordings $data/reco2dur
    true
  fi
}

function filter_speakers {
  # throughout this program, we regard utt2spk as primary and spk2utt as derived, so...
  utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt

  cat $data/spk2utt | awk '{print $1}' > $tmpdir/speakers
  for s in cmvn.scp spk2gender; do
    f=$data/$s
    if [ -f $f ]; then
      filter_file $f $tmpdir/speakers
    fi
  done

  filter_file $tmpdir/speakers $data/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/spk2utt > $data/utt2spk

  for s in cmvn.scp spk2gender $spk_extra_files; do
    f=$data/$s
    if [ -f $f ]; then
      filter_file $tmpdir/speakers $f
    fi
  done
}

function filter_utts {
  cat $data/utt2spk | awk '{print $1}' > $tmpdir/utts

  ! cat $data/utt2spk | sort | cmp - $data/utt2spk && \
    echo "utt2spk is not in sorted order (fix this yourself)" && exit 1;

  ! cat $data/utt2spk | sort -k2 | cmp - $data/utt2spk && \
    echo "utt2spk is not in sorted order when sorted first on speaker-id " && \
    echo "(fix this by making speaker-ids prefixes of utt-ids)" && exit 1;

  ! cat $data/spk2utt | sort | cmp - $data/spk2utt && \
    echo "spk2utt is not in sorted order (fix this yourself)" && exit 1;

  if [ -f $data/utt2uniq ]; then
    ! cat $data/utt2uniq | sort | cmp - $data/utt2uniq && \
      echo "utt2uniq is not in sorted order (fix this yourself)" && exit 1;
  fi

  maybe_wav=
  maybe_reco2dur=
  [ ! -f $data/segments ] && maybe_wav=wav.scp # wav indexed by utts only if segments does not exist.
  [ -s $data/reco2dur ] && [ ! -f $data/segments ] && maybe_reco2dur=reco2dur # reco2dur indexed by utts

  maybe_utt2dur=
  if [ -f $data/utt2dur ]; then
    cat $data/utt2dur | \
      awk '{ if (NF == 2 && $2 > 0) { print }}' > $data/utt2dur.ok || exit 1
    maybe_utt2dur=utt2dur.ok
  fi

  maybe_utt2num_frames=
  if [ -f $data/utt2num_frames ]; then
    cat $data/utt2num_frames | \
      awk '{ if (NF == 2 && $2 > 0) { print }}' > $data/utt2num_frames.ok || exit 1
    maybe_utt2num_frames=utt2num_frames.ok
  fi

  for x in feats.scp text segments utt2lang $maybe_wav $maybe_utt2dur $maybe_utt2num_frames; do
    if [ -f $data/$x ]; then
      utils/filter_scp.pl $data/$x $tmpdir/utts > $tmpdir/utts.tmp
      mv $tmpdir/utts.tmp $tmpdir/utts
    fi
  done
  rm $data/utt2dur.ok 2>/dev/null || true
  rm $data/utt2num_frames.ok 2>/dev/null || true

  [ ! -s $tmpdir/utts ] && echo "fix_data_dir.sh: no utterances remained: not proceeding further." && \
    rm $tmpdir/utts && exit 1;


  if [ -f $data/utt2spk ]; then
    new_nutts=$(cat $tmpdir/utts | wc -l)
    old_nutts=$(cat $data/utt2spk | wc -l)
    if [ $new_nutts -ne $old_nutts ]; then
      echo "fix_data_dir.sh: kept $new_nutts utterances out of $old_nutts"
    else
      echo "fix_data_dir.sh: kept all $old_nutts utterances."
    fi
  fi

  for x in utt2spk utt2uniq feats.scp vad.scp text segments utt2lang utt2dur utt2num_frames $maybe_wav $maybe_reco2dur $utt_extra_files; do
    if [ -f $data/$x ]; then
      cp $data/$x $data/.backup/$x
      if ! cmp -s $data/$x <( utils/filter_scp.pl $tmpdir/utts $data/$x ) ; then
        utils/filter_scp.pl $tmpdir/utts $data/.backup/$x > $data/$x
      fi
    fi
  done

}

filter_recordings
filter_speakers
filter_utts
filter_speakers
filter_recordings

utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt

echo "fix_data_dir.sh: old files are kept in $data/.backup"
