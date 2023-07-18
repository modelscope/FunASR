#!/bin/bash

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita, Shota Horiguchi)
# Licensed under the MIT license.
#
# This script prepares kaldi-style data sets shared with different experiments
#   - data/xxxx
#     callhome, sre, swb2, and swb_cellular datasets
#   - data/simu_${simu_outputs}
#     simulation mixtures generated with various options

stage=0

# Modify corpus directories
#  - callhome_dir
#    CALLHOME (LDC2001S97)
#  - swb2_phase1_train
#    Switchboard-2 Phase 1 (LDC98S75)
#  - data_root
#    LDC99S79, LDC2002S06, LDC2001S13, LDC2004S07,
#    LDC2006S44, LDC2011S01, LDC2011S04, LDC2011S09,
#    LDC2011S10, LDC2012S01, LDC2011S05, LDC2011S08
#  - musan_root
#    MUSAN corpus (https://www.openslr.org/17/)
callhome_dir=/export/corpora/NIST/LDC2001S97
swb2_phase1_train=/export/corpora/LDC/LDC98S75
data_root=/export/corpora5/LDC
musan_root=/export/corpora/JHU/musan
# Modify simulated data storage area.
# This script distributes simulated data under these directories
simu_actual_dirs=(
/export/c05/$USER/diarization-data
/export/c08/$USER/diarization-data
/export/c09/$USER/diarization-data
)

# data preparation options
max_jobs_run=4
sad_num_jobs=30
sad_opts="--extra-left-context 79 --extra-right-context 21 --frames-per-chunk 150 --extra-left-context-initial 0 --extra-right-context-final 0 --acwt 0.3"
sad_graph_opts="--min-silence-duration=0.03 --min-speech-duration=0.3 --max-speech-duration=10.0"
sad_priors_opts="--sil-scale=0.1"

# simulation options
simu_opts_overlap=yes
simu_opts_num_speaker_array=(1 2 3 4)
simu_opts_sil_scale_array=(2 2 5 9)
simu_opts_rvb_prob=0.5
simu_opts_num_train=100000
simu_opts_min_utts=10
simu_opts_max_utts=20

simu_cmd="run.pl"
train_cmd="run.pl"
random_mixture_cmd="run.pl"
make_mixture_cmd="run.pl"

. parse_options.sh || exit

if [ $stage -le 0 ]; then
    echo "prepare kaldi-style datasets"
    # Prepare CALLHOME dataset. This will be used to evaluation.
    if ! validate_data_dir.sh --no-text --no-feats data/callhome1_spkall \
        || ! validate_data_dir.sh --no-text --no-feats data/callhome2_spkall; then
        # imported from https://github.com/kaldi-asr/kaldi/blob/master/egs/callhome_diarization/v1
        local/make_callhome.sh $callhome_dir data
        # Generate two-speaker subsets
        for dset in callhome1 callhome2; do
            # Extract two-speaker recordings in wav.scp
            copy_data_dir.sh data/${dset} data/${dset}_spkall
            # Regenerate segments file from fullref.rttm
            #  $2: recid, $4: start_time, $5: duration, $8: speakerid
            awk '{printf "%s_%s_%07d_%07d %s %.2f %.2f\n", \
                 $2, $8, $4*100, ($4+$5)*100, $2, $4, $4+$5}' \
                data/callhome/fullref.rttm | sort > data/${dset}_spkall/segments
            utils/fix_data_dir.sh data/${dset}_spkall
            # Speaker ID is '[recid]_[speakerid]
            awk '{split($1,A,"_"); printf "%s %s_%s\n", $1, A[1], A[2]}' \
                data/${dset}_spkall/segments > data/${dset}_spkall/utt2spk
            utils/fix_data_dir.sh data/${dset}_spkall
            # Generate rttm files for scoring
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                data/${dset}_spkall/utt2spk data/${dset}_spkall/segments \
                data/${dset}_spkall/rttm
            utils/data/get_reco2dur.sh data/${dset}_spkall
        done
    fi
    # Prepare a collection of NIST SRE and SWB data. This will be used to train,
    if ! validate_data_dir.sh --no-text --no-feats data/swb_sre_comb; then
        local/make_sre.sh $data_root data
        # Prepare SWB for x-vector DNN training.
        local/make_swbd2_phase1.pl $swb2_phase1_train \
            data/swbd2_phase1_train
        local/make_swbd2_phase2.pl $data_root/LDC99S79 \
            data/swbd2_phase2_train
        local/make_swbd2_phase3.pl $data_root/LDC2002S06 \
            data/swbd2_phase3_train
        local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
            data/swbd_cellular1_train
        local/make_swbd_cellular2.pl $data_root/LDC2004S07 \
            data/swbd_cellular2_train
        # Combine swb and sre data
        utils/combine_data.sh data/swb_sre_comb \
            data/swbd_cellular1_train data/swbd_cellular2_train \
            data/swbd2_phase1_train \
            data/swbd2_phase2_train data/swbd2_phase3_train data/sre
    fi
    # musan data. "back-ground
    if ! validate_data_dir.sh --no-text --no-feats data/musan_noise_bg; then
        local/make_musan.sh $musan_root data
        utils/copy_data_dir.sh data/musan_noise data/musan_noise_bg
        awk '{if(NR>1) print $1,$1}'  $musan_root/noise/free-sound/ANNOTATIONS > data/musan_noise_bg/utt2spk
        utils/fix_data_dir.sh data/musan_noise_bg
    fi
    # simu rirs 8k
    if ! validate_data_dir.sh --no-text --no-feats data/simu_rirs_8k; then
        mkdir -p data/simu_rirs_8k
        if [ ! -e sim_rir_8k.zip ]; then
            wget --no-check-certificate http://www.openslr.org/resources/26/sim_rir_8k.zip
        fi
        unzip sim_rir_8k.zip -d data/sim_rir_8k
        find $PWD/data/sim_rir_8k -iname "*.wav" \
            | awk '{n=split($1,A,/[\/\.]/); print A[n-3]"_"A[n-1], $1}' \
            | sort > data/simu_rirs_8k/wav.scp
        awk '{print $1, $1}' data/simu_rirs_8k/wav.scp > data/simu_rirs_8k/utt2spk
        utils/fix_data_dir.sh data/simu_rirs_8k
    fi
    # Automatic segmentation using pretrained SAD model
    #     it will take one day using 30 CPU jobs:
    #     make_mfcc: 1 hour, compute_output: 18 hours, decode: 0.5 hours
    sad_nnet_dir=exp/segmentation_1a/tdnn_stats_asr_sad_1a
    sad_work_dir=exp/segmentation_1a/tdnn_stats_asr_sad_1a
    if ! validate_data_dir.sh --no-text $sad_work_dir/swb_sre_comb_seg; then
        if [ ! -d exp/segmentation_1a ]; then
            wget http://kaldi-asr.org/models/4/0004_tdnn_stats_asr_sad_1a.tar.gz
            tar zxf 0004_tdnn_stats_asr_sad_1a.tar.gz
        fi
        steps/segmentation/detect_speech_activity.sh \
            --nj $sad_num_jobs \
            --graph-opts "$sad_graph_opts" \
            --transform-probs-opts "$sad_priors_opts" $sad_opts \
            data/swb_sre_comb $sad_nnet_dir mfcc_hires $sad_work_dir \
            $sad_work_dir/swb_sre_comb || exit 1
    fi
    # Extract >1.5 sec segments and split into train/valid sets
    if ! validate_data_dir.sh --no-text --no-feats data/swb_sre_cv; then
        copy_data_dir.sh data/swb_sre_comb data/swb_sre_comb_seg
        awk '$4-$3>1.5{print;}' $sad_work_dir/swb_sre_comb_seg/segments > data/swb_sre_comb_seg/segments
        cp $sad_work_dir/swb_sre_comb_seg/{utt2spk,spk2utt} data/swb_sre_comb_seg
        fix_data_dir.sh data/swb_sre_comb_seg
        utils/subset_data_dir_tr_cv.sh data/swb_sre_comb_seg data/swb_sre_tr data/swb_sre_cv
    fi
fi

simudir=data/simu
if [ $stage -le 1 ]; then
    echo "simulation of mixture"
    mkdir -p $simudir/.work
    local/random_mixture_cmd=random_mixture.py
    local/make_mixture_cmd=make_mixture.py

    for ((i=0; i<${#simu_opts_sil_scale_array[@]}; ++i)); do
        simu_opts_num_speaker=${simu_opts_num_speaker_array[i]}
        simu_opts_sil_scale=${simu_opts_sil_scale_array[i]}
        for dset in swb_sre_tr swb_sre_cv; do
            if [ "$dset" == "swb_sre_tr" ]; then
                n_mixtures=${simu_opts_num_train}
            else
                n_mixtures=500
            fi
            simuid=${dset}_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_${n_mixtures}
            # check if you have the simulation
            if ! validate_data_dir.sh --no-text --no-feats $simudir/data/$simuid; then
                # random mixture generation
                $train_cmd $simudir/.work/random_mixture_$simuid.log \
                    $random_mixture_cmd --n_speakers $simu_opts_num_speaker --n_mixtures $n_mixtures \
                    --speech_rvb_probability $simu_opts_rvb_prob \
                    --sil_scale $simu_opts_sil_scale \
                    data/$dset data/musan_noise_bg data/simu_rirs_8k \
                    \> $simudir/.work/mixture_$simuid.scp
                nj=64
                mkdir -p $simudir/wav/$simuid
                # distribute simulated data to $simu_actual_dir
                split_scps=
                for n in $(seq $nj); do
                    split_scps="$split_scps $simudir/.work/mixture_$simuid.$n.scp"
                    mkdir -p $simudir/.work/data_$simuid.$n
                    actual=${simu_actual_dirs[($n-1)%${#simu_actual_dirs[@]}]}/$simudir/wav/$simuid/$n
                    mkdir -p $actual
                    ln -nfs $actual $simudir/wav/$simuid/$n
                done
                utils/split_scp.pl $simudir/.work/mixture_$simuid.scp $split_scps || exit 1

                $simu_cmd --max-jobs-run 64 JOB=1:$nj $simudir/.work/make_mixture_$simuid.JOB.log \
                    $make_mixture_cmd --rate=8000 \
                    $simudir/.work/mixture_$simuid.JOB.scp \
                    $simudir/.work/data_$simuid.JOB $simudir/wav/$simuid/JOB
                utils/combine_data.sh $simudir/data/$simuid $simudir/.work/data_$simuid.*
                steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    $simudir/data/$simuid/utt2spk $simudir/data/$simuid/segments \
                    $simudir/data/$simuid/rttm
                utils/data/get_reco2dur.sh $simudir/data/$simuid
            fi
            simuid_concat=${dset}_ns"$(IFS="n"; echo "${simu_opts_num_speaker_array[*]}")"_beta"$(IFS="n"; echo "${simu_opts_sil_scale_array[*]}")"_${n_mixtures}
            mkdir -p $simudir/data/$simuid_concat
            for f in `ls -F $simudir/data/$simuid | grep -v "/"`; do
                cat $simudir/data/$simuid/$f >> $simudir/data/$simuid_concat/$f
            done
        done
    done
fi

if [ $stage -le 3 ]; then
    # compose eval/callhome2_spkall
    eval_set=data/eval/callhome2_spkall
    if ! validate_data_dir.sh --no-text --no-feats $eval_set; then
        utils/copy_data_dir.sh data/callhome2_spkall $eval_set
        cp data/callhome2_spkall/rttm $eval_set/rttm
        awk -v dstdir=wav/eval/callhome2_spkall '{print $1, dstdir"/"$1".wav"}' data/callhome2_spkall/wav.scp > $eval_set/wav.scp
        mkdir -p wav/eval/callhome2_spkall
        wav-copy scp:data/callhome2_spkall/wav.scp scp:$eval_set/wav.scp
        utils/data/get_reco2dur.sh $eval_set
    fi

    # compose eval/callhome1_spkall
    adapt_set=data/eval/callhome1_spkall
    if ! validate_data_dir.sh --no-text --no-feats $adapt_set; then
        utils/copy_data_dir.sh data/callhome1_spkall $adapt_set
        cp data/callhome1_spkall/rttm $adapt_set/rttm
        awk -v dstdir=wav/eval/callhome1_spkall '{print $1, dstdir"/"$1".wav"}' data/callhome1_spkall/wav.scp > $adapt_set/wav.scp
        mkdir -p wav/eval/callhome1_spkall
        wav-copy scp:data/callhome1_spkall/wav.scp scp:$adapt_set/wav.scp
        utils/data/get_reco2dur.sh $adapt_set
    fi
fi
