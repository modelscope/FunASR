#!/usr/bin/env bash

. ./path.sh || exit 1;
# Begin configuration section.
nj=32
cmd=./utils/run.pl

# feature configuration
feat_dims=80
sample_frequency=16000
speed_perturb="1.0"

echo "$0 $@"

. utils/parse_options.sh || exit 1;

data=$1
logdir=$2
fbankdir=$3

[ ! -f $data/wav.scp ] && echo "$0: no such file $data/wav.scp" && exit 1;
[ ! -f $data/text ] && echo "$0: no such file $data/text" && exit 1;

python utils/split_data.py $data $data $nj

ark_dir=${fbankdir}/ark; mkdir -p ${ark_dir}
text_dir=${fbankdir}/txt; mkdir -p ${text_dir}
mkdir -p ${logdir}

$cmd JOB=1:$nj $logdir/make_fbank.JOB.log \
    python utils/compute_fbank.py -w $data/split${nj}/JOB/wav.scp -t $data/split${nj}/JOB/text \
        -d $feat_dims -s $sample_frequency -p ${speed_perturb} -a JOB -o ${fbankdir} \
        || exit 1;

for n in $(seq $nj); do
    cat ${ark_dir}/feats.$n.scp || exit 1
done > $fbankdir/feats.scp || exit 1

for n in $(seq $nj); do
    cat ${text_dir}/text.$n.txt || exit 1
done > $fbankdir/text || exit 1

for n in $(seq $nj); do
    cat ${ark_dir}/len.$n || exit 1
done > $fbankdir/speech_shape || exit 1

for n in $(seq $nj); do
    cat ${text_dir}/len.$n || exit 1
done > $fbankdir/text_shape || exit 1

echo "$0: Succeeded compute FBANK features"
