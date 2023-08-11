#!/usr/bin/env bash

. ./path.sh || exit 1;

# This recipe aims at reimplement the results of SOND on Callhome corpus which is represented in
# [1] TOLD: A Novel Two-stage Overlap-aware Framework for Speaker Diarization, ICASSP 2023
# You can also use it on other dataset such AliMeeting to reproduce the results in
# [2] Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis, EMNLP 2022
# We recommend you run this script stage by stage.

# This recipe includes:
# 1. simulating data with switchboard and NIST.
# 2. training the model from scratch for 3 stages:
#   2-1. pre-train on simu_swbd_sre
#   2-2. train on simu_swbd_sre
#   2-3. finetune on callhome1
# 3. evaluating model with the results from the first stage EEND-OLA,
# Finally, you will get a similar DER result claimed in the paper.

# environment configuration
# path/to/kaldi
kaldi_root=

if [ -z "${kaldi_root}" ]; then
  echo "We need kaldi to prepare dataset, extract fbank features, please install kaldi first and set kaldi_root."
  echo "Kaldi installation guide can be found at https://kaldi-asr.org/"
  exit;
fi

if [ ! -e local ]; then
  ln -s ${kaldi_root}/egs/callhome_diarization/v2/local ./local
fi

if [ ! -e utils ]; then
  ln -s ${kaldi_root}/egs/callhome_diarization/v2/utils ./utils
fi

# path to Switchboard and NIST including:
# LDC98S75, LDC99S79, LDC2002S06, LDC2001S13, LDC2004S07
data_root=
if [ -z "${data_root}" ]; then
  echo "We need Switchboard and NIST to simulate data for pretraining."
  echo "If you can't get them, please use 'finetune.sh' to finetune a pretrained model."
  exit;
fi

# path/to/NIST/LDC2001S97
callhome_root=
if [ -z "${callhome_root}" ]; then
  echo "We need callhome corpus for training."
  echo "If you want inference only, please refer https://www.modelscope.cn/models/damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch/summary"
  exit;
fi


# machines configuration
gpu_devices="4,5,6,7"  # for V100-16G, use 4 GPUs
gpu_num=4
count=1

# general configuration
stage=0
stop_stage=19
# number of jobs for data process
nj=16
sr=8000

# experiment configuration
lang=en
feats_type=fbank
datadir=data
dumpdir=dump
expdir=exp
train_cmd=utils/run.pl

# training related
tag=""
train_set=simu_swbd_sre
valid_set=callhome1
train_config=conf/EAND_ResNet34_SAN_L4N512_None_FFN_FSMN_L6N512_bce_dia_loss_01.yaml
token_list=${datadir}/token_list/powerset_label_n16k4.txt
init_param=
freeze_param=

# inference related
inference_model=valid.der.ave_5best.pb
inference_config=conf/basic_inference.yaml
inference_tag=""
test_sets="callhome2"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# number of jobs for inference
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=4
infer_cmd=utils/run.pl
told_max_iter=4

. utils/parse_options.sh || exit 1;

model_dir="$(basename "${train_config}" .yaml)_${feats_type}_${lang}${tag}"

# you can set gpu num for decoding here
gpuid_list=$gpu_devices  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi

# Prepare datasets
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # 1. Prepare a collection of NIST SRE data.
  echp "Stage 0: Prepare a collection of NIST SRE data."

  local/make_sre.sh $data_root ${datadir}

  # 2.a Prepare SWB.
  local/make_swbd2_phase1.pl ${data_root}/LDC98S75 \
    ${datadir}/swbd2_phase1_train
  local/make_swbd2_phase2.pl $data_root/LDC99S79 \
                           ${datadir}/swbd2_phase2_train
  local/make_swbd2_phase3.pl $data_root/LDC2002S06 \
                           ${datadir}/swbd2_phase3_train
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
                             ${datadir}/swbd_cellular1_train
  local/make_swbd_cellular2.pl $data_root/LDC2004S07 \
                             ${datadir}/swbd_cellular2_train
  # 2.b combine all swbd data.
  utils/combine_data.sh ${datadir}/swbd \
      ${datadir}/swbd2_phase1_train ${datadir}/swbd2_phase2_train ${datadir}/swbd2_phase3_train \
      ${datadir}/swbd_cellular1_train ${datadir}/swbd_cellular2_train
  utils/validate_data_dir.sh --no-text --no-feats ${datadir}/swbd
  utils/fix_data_dir.sh ${datadir}/swbd

  utils/combine_data.sh ${datadir}/swbd_sre ${datadir}/swbd ${datadir}/sre
  utils/validate_data_dir.sh --no-text --no-feats ${datadir}/swbd_sre
  utils/fix_data_dir.sh ${datadir}/swbd_sre

  # 3. Prepare the Callhome portion of NIST SRE 2000.
  local/make_callhome.sh ${callhome_root} ${datadir}/

  # 4. split ref.rttm
  for dset in callhome1 callhome2; do
    rm -rf ${datadir}/${dset}/ref.rttm
    for name in `awk '{print $1}' ${datadir}/${dset}/wav.scp`; do
      grep ${name} ${datadir}/callhome/fullref.rttm >> ${datadir}/${dset}/ref.rttm;
    done

    # filter out records which don't have rttm labels.
    awk '{print $2}' ${datadir}/${dset}/ref.rttm | sort | uniq > ${datadir}/${dset}/uttid
    mv ${datadir}/${dset}/wav.scp ${datadir}/${dset}/wav.scp.bak
    awk '{if (NR==FNR){a[$1]=1}else{if (a[$1]==1){print $0}}}' ${datadir}/${dset}/uttid ${datadir}/${dset}/wav.scp.bak > ${datadir}/${dset}/wav.scp
    mkdir ${datadir}/${dset}/raw
    mv ${datadir}/${dset}/{reco2num_spk,segments,spk2utt,utt2spk,uttid,wav.scp.bak} ${datadir}/${dset}/raw/
    awk '{print $1,$1}' ${datadir}/${dset}/wav.scp > ${datadir}/${dset}/utt2spk
  done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Dump sph file to wav"
  export PATH=${kaldi_root}/tools/sph2pipe/:${PATH}
  if [ ! -f ${kaldi_root}/tools/sph2pipe/sph2pipe ]; then
    echo "Can not find sph2pipe in ${kaldi_root}/tools/sph2pipe/,"
    echo "please install sph2pipe and put it in the right place."
    exit;
  fi

  for dset in callhome1 callhome2 swbd_sre; do
    echo "Stage 1: start to dump ${dset}."
    mv ${datadir}/${dset}/wav.scp ${datadir}/${dset}/sph.scp

    mkdir -p ${dumpdir}/${dset}/wavs
    python -Wignore script/dump_pipe_wav.py ${datadir}/${dset}/sph.scp ${dumpdir}/${dset}/wavs \
      --sr ${sr} --nj ${nj} --no_pbar
    find `pwd`/${dumpdir}/${dset}/wavs -iname "*.wav" | sort | awk -F'[/.]' '{print $(NF-1),$0}' > ${datadir}/${dset}/wav.scp
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Extract non-overlap segments from callhome dataset"
  for dset in callhome1 callhome2; do
    echo "Stage 2: Extracting non-overlap segments for "${dset}
    mkdir -p ${dumpdir}/${dset}/nonoverlap_0s
    python -Wignore script/extract_nonoverlap_segments.py \
      ${datadir}/${dset}/wav.scp ${datadir}/${dset}/ref.rttm ${dumpdir}/${dset}/nonoverlap_0s \
      --min_dur 0.1 --max_spk_num 8 --sr ${sr} --no_pbar --nj ${nj}

    mkdir -p ${datadir}/${dset}/nonoverlap_0s
    find ${dumpdir}/${dset}/nonoverlap_0s/ -iname "*.wav" | sort | awk -F'[/.]' '{print $(NF-1),$0}' > ${datadir}/${dset}/nonoverlap_0s/wav.scp
    awk -F'[/.]' '{print $(NF-1),$(NF-2)}' ${datadir}/${dset}/nonoverlap_0s/wav.scp > ${datadir}/${dset}/nonoverlap_0s/utt2spk
    echo "Done."
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Generate concatenated waveforms for each speaker in switchboard, sre and callhome1"
  mkdir swb_sre_resources
  wget --no-check-certificate -P swb_sre_resources/ https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/Speaker_Diar/swb_sre_resources/noise.scp
  wget --no-check-certificate -P swb_sre_resources/ https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/Speaker_Diar/swb_sre_resources/swbd_sre_tdnn_vad_segments
  mkdir ${datadir}/swbd_sre/none_silence
  ln -s swb_sre_resources/swbd_sre_tdnn_vad_segments ${datadir}/swbd_sre/none_silence/segments
  cp ${datadir}/swbd_sre/wav.scp ${datadir}/swbd_sre/none_silence/reco.scp

  mkdir -p ${dumpdir}/swbd_sre/none_silence
  python -Wignore script/remove_silence_from_wav.py \
    ${datadir}/swbd_sre/none_silence ${dumpdir}/swbd_sre/none_silence --nj ${nj} --sr 8000
  # The utterance number in wav.scp may be different from reco.scp,
  # since some recordings don't appear in the segments file, may due to the VAD
  echo "find wavs_nosil"
  find `pwd`/${dumpdir}/swbd_sre/none_silence -iname "*.wav" | sort | awk -F'[/.]' '{print $(NF-1),$0}' > ${datadir}/swbd_sre/none_silence/wav.scp
  echo "concat spk segments"

  ln -s ${datadir}/swbd_sre/utt2spk ${datadir}/swbd_sre/none_silence/utt2spk

  echo "Stage 3: Start to concatnate waveforms for speakers in switchboard and sre"
  python -Wignore egs/callhome/concat_spk_segs.py \
    ${datadir}/swbd_sre/none_silence ${dumpdir}/swbd_sre/spk_wavs --nj ${nj} --sr 8000

  echo "Stage 3: Start to concatnate waveforms for speakers in callhome1"
  # only use callhome1 as training set to simulate data
  python -Wignore egs/callhome/concat_spk_segs.py \
    ${datadir}/callhome1/nonoverlap_0s ${dumpdir}/callhome1/spk_wavs --nj ${nj} --sr 8000

fi

# simulate data with the pattern of callhome1
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Start to simulate recordings."

  if [ ! -e ${dumpdir}/musan ]; then
    echo "Stage 4-1: Start to download MUSAN noises from openslr"
    wget --no-check-certificate -P ${dumpdir}/musan https://www.openslr.org/resources/17/musan.tar.gz
    tar -C ${dumpdir}/musan -xvf ${dumpdir}/musan/musan.tar.gz
  fi

  if [ ! -e ${dumpdir}/rirs ]; then
    echo "Stage 4-2: Start to download RIRs from openslr"
    wget --no-check-certificate -P ${dumpdir}/rirs https://www.openslr.org/resources/28/rirs_noises.zip
    unzip ${dumpdir}/rirs/rirs_noises.zip -d ${dumpdir}/rirs
  fi

  mkdir -p ${datadir}/simu_swbd_sre
  # only use background noises instead of all noises in MUSAN.
  sed "s:/path/to/musan/:`pwd`/${dumpdir}/musan/:g" swb_sre_resources/noise.scp > ${datadir}/simu_swbd_sre/noise.scp
  # use simulated RIRs.
  find `pwd`/${dumpdir}/rirs/RIRS_NOISES/simulated_rirs/ -iname "*.wav" | sort | awk -F'[/.]' '{print $(NF-3)"-"$(NF-1), $0}' > ${datadir}/simu_swbd_sre/rirs.scp
  cp ${datadir}/callhome1/{ref.rttm,reco2num_spk} ${datadir}/simu_swbd_sre
  find `pwd`/${dumpdir}/swbd_sre/spk_wavs -iname "*.wav" | sort | awk -F'[/.]' '{print $(NF-1),$0}' > ${datadir}/simu_swbd_sre/spk2wav.scp

  echo "Stage 4-3: Start to simulate recordings with variable speakers as Callhome1 patterns."
  # average duration of callhome is 125s, about 20 chunk with silence
  # simulating 22500 (45 jobs x 500 reco) recordings, without random_assign and random_shift_interval
  for i in $(seq 0 44); do
    cmd="python -Wignore egs/callhome/simu_whole_recordings.py \
           ${datadir}/simu_swbd_sre \
           ${dumpdir}/simu_swbd_sre/wavs \
          --corpus_name simu_swbd_sre --task_id $i --total_mix 500 --sr 8000 --no_bar &"
    echo $cmd
    eval $cmd
  done
  wait;

  echo "Stage 4-4: Start to simulate recordings with fixed speakers as Callhome1 patterns."
  # simulating 30000 (30 jobs x 1000 reco) recordings for different speaker number 2, 3, 4
  for n_spk in $(seq 2 4); do
    mkdir -p /home/neo.dzh/corpus/simu_swbd_sre/${n_spk}spk_wavs
    for i in $(seq 0 29); do
      cmd="python -Wignore egs/callhome/simu_whole_recordings.py \
             ${datadir}/simu_swbd_sre \
             ${dumpdir}/simu_swbd_sre/${n_spk}spk_wavs \
             --random_assign_spk --random_interval --spk_num ${n_spk} \
             --corpus_name simu_swbd_sre --task_id $i --total_mix 1000 --sr 8000 --no_bar &"
      echo $cmd
      eval $cmd
    done
    wait;
  done

  find `pwd`/${dumpdir}/simu_swbd_sre -iname "*.wav" | sort | awk -F'[/.]' '{print $(NF-1),$0}' > ${datadir}/simu_swbd_sre/wav.scp
  awk '{print $1,$1}' ${datadir}/simu_swbd_sre/wav.scp > ${datadir}/simu_swbd_sre/utt2spk
  find `pwd`/${dumpdir}/simu_swbd_sre -iname "*.rttm" | sort | awk -F'[/.]' '{print $(NF-1),$0}' > ${datadir}/simu_swbd_sre/rttm.scp
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: Generate fbank features"
  home_path=`pwd`
  cd ${kaldi_root}/egs/callhome_diarization/v2 || exit

  . ./cmd.sh
  . ./path.sh

  for dset in simu_swbd_sre callhome1 callhome2; do
    steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf --nj ${nj} --cmd "$train_cmd" \
        ${datadir}/${dset} ${expdir}/make_fbank/${dset} ${dumpdir}/${dset}/fbank
    utils/fix_data_dir.sh ${datadir}/${dset}
  done

  for dset in swbd_sre/none_silence callhome1/nonoverlap_0s callhome2/nonoverlap_0s; do
    steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf --nj ${nj} --cmd "$train_cmd" \
        ${datadir}/${dset} ${expdir}/make_fbank/${dset} ${dumpdir}/${dset}/fbank
    utils/fix_data_dir.sh ${datadir}/${dset}
  done

  cd ${home_path} || exit
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: Extract speaker embeddings."
  sv_exp_dir=exp/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch

  if [ ! -e ${sv_exp_dir} ]; then
    echo "start to download sv models"
    git lfs install
    git clone https://www.modelscope.cn/damo/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch.git
    mv speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch ${expdir}/
    echo "Done."
  fi

  sed "s/input_size: null/input_size: 80/g" ${sv_exp_dir}/sv.yaml > ${sv_exp_dir}/sv_fbank.yaml
  for dset in swbd_sre/none_silence callhome1/nonoverlap_0s callhome2/nonoverlap_0s; do
    key_file=${datadir}/${dset}/feats.scp
    num_scp_file="$(<${key_file} wc -l)"
    _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
    _logdir=${dumpdir}/${dset}/xvecs
    mkdir -p ${_logdir}
    split_scps=
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/sv_inference.JOB.log \
      python -m funasr.bin.sv_inference_launch \
        --batch_size 1 \
        --njob ${njob} \
        --ngpu "${_ngpu}" \
        --gpuid_list ${gpuid_list} \
        --data_path_and_name_and_type "${key_file},speech,kaldi_ark" \
        --key_file "${_logdir}"/keys.JOB.scp \
        --sv_train_config ${sv_exp_dir}/sv_fbank.yaml \
        --sv_model_file ${sv_exp_dir}/sv.pth \
        --output_dir "${_logdir}"/output.JOB
    cat ${_logdir}/output.*/xvector.scp | sort > ${datadir}/${dset}/utt2xvec
  done

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7: Generate label files."

  for dset in simu_swbd_sre callhome1 callhome2; do
    echo "Stage 7: Generate labels for ${dset}."
    python -Wignore script/calc_real_meeting_frame_labels.py \
          ${datadir}/${dset} ${dumpdir}/${dset}/labels \
          --n_spk 8 --frame_shift 0.01 --nj 16 --sr 8000
    find `pwd`/${dumpdir}/${dset}/labels/ -iname "*.lbl.mat" | awk -F'[/.]' '{print $(NF-2),$0}' | sort > ${datadir}/${dset}/labels.scp
  done

fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Stage 8: Make training and evaluation files."

  # dump simulated data in training mode (randomly shuffle the speaker order).
  data_dir=${datadir}/simu_swbd_sre/files_for_dump
  mkdir ${data_dir}
  cp ${datadir}/simu_swbd_sre/{feats.scp,labels.scp} ${data_dir}/
  cp ${datadir}/swbd_sre/none_silence/{utt2spk,utt2xvec,utt2num_frames} ${data_dir}/
  # dump data with the window length of 1600 frames and hop length of 400 frames.
  echo "Stage 8: start to dump for simu_swbd_sre."
  for i in $(seq 0 49); do
    cmd="python -Wignore script/dump_meeting_chunks.py --dir ${data_dir} \
      --out ${dumpdir}/simu_swbd_sre/dumped_files/data --n_spk 16 --no_pbar --sr 8000 --mode train \
      --chunk_size 1600 --chunk_shift 400 \
      --task_id ${i} --task_size 2250 &"
    echo $cmd
    eval $cmd
  done
  wait;
  mkdir -p ${datadir}/simu_swbd_sre/dumped_files
  cat ${dumpdir}/simu_swbd_sre/dumped_files/data_parts*_feat.scp | sort > ${datadir}/simu_swbd_sre/dumped_files/feats.scp
  cat ${dumpdir}/simu_swbd_sre/dumped_files/data_parts*_xvec.scp | sort > ${datadir}/simu_swbd_sre/dumped_files/profile.scp
  cat ${dumpdir}/simu_swbd_sre/dumped_files/data_parts*_label.scp | sort > ${datadir}/simu_swbd_sre/dumped_files/label.scp
  mkdir -p ${expdir}/simu_swbd_sre_states
  awk '{print $1,"1600"}' ${datadir}/simu_swbd_sre/dumped_files/feats.scp | shuf > ${expdir}/simu_swbd_sre_states/speech_shape

  # dump callhome1 data in training mode.
  data_dir=${datadir}/callhome1/files_for_dump
  mkdir ${data_dir}
  # filter out zero duration segments
  LC_ALL=C awk '{if ($5 > 0){print $0}}' ${datadir}/callhome1/ref.rttm > ${data_dir}/ref.rttm
  cp ${datadir}/callhome1/{feats.scp,labels.scp} ${data_dir}/
  cp ${datadir}/callhome1/nonoverlap_0s/{utt2spk,utt2xvec,utt2num_frames} ${data_dir}/

  echo "Stage 8: start to dump for callhome1."
  python -Wignore script/dump_meeting_chunks.py --dir ${data_dir} \
    --out ${dumpdir}/callhome1/dumped_files/data --n_spk 16 --no_pbar --sr 8000 --mode train \
    --chunk_size 1600 --chunk_shift 400 --add_mid_to_speaker true

  mkdir -p ${datadir}/callhome1/dumped_files
  cat ${dumpdir}/callhome1/dumped_files/data_parts*_feat.scp | sort > ${datadir}/callhome1/dumped_files/feats.scp
  cat ${dumpdir}/callhome1/dumped_files/data_parts*_xvec.scp | sort > ${datadir}/callhome1/dumped_files/profile.scp
  cat ${dumpdir}/callhome1/dumped_files/data_parts*_label.scp | sort > ${datadir}/callhome1/dumped_files/label.scp
  mkdir -p ${expdir}/callhome1_states
  awk '{print $1,"1600"}' ${datadir}/callhome1/dumped_files/feats.scp | shuf > ${expdir}/callhome1_states/speech_shape
  python -Wignore script/convert_rttm_to_seg_file.py --rttm_scp ${data_dir}/ref.rttm --seg_file ${data_dir}/org_vad.txt

  # dump callhome2 data in test mode.
  data_dir=${datadir}/callhome2/files_for_dump
  mkdir ${data_dir}
  # filter out zero duration segments
  LC_ALL=C awk '{if ($5 > 0){print $0}}' ${datadir}/callhome2/ref.rttm > ${data_dir}/ref.rttm
  cp ${datadir}/callhome2/{feats.scp,labels.scp} ${data_dir}/
  cp ${datadir}/callhome2/nonoverlap_0s/{utt2spk,utt2xvec,utt2num_frames} ${data_dir}/

  echo "Stage 8: start to dump for callhome2."
  python -Wignore script/dump_meeting_chunks.py --dir ${data_dir} \
    --out ${dumpdir}/callhome2/dumped_files/data --n_spk 16 --no_pbar --sr 8000 --mode test \
    --chunk_size 1600 --chunk_shift 400 --add_mid_to_speaker true

  mkdir -p ${datadir}/callhome2/dumped_files
  cat ${dumpdir}/callhome2/dumped_files/data_parts*_feat.scp | sort > ${datadir}/callhome2/dumped_files/feats.scp
  cat ${dumpdir}/callhome2/dumped_files/data_parts*_xvec.scp | sort > ${datadir}/callhome2/dumped_files/profile.scp
  cat ${dumpdir}/callhome2/dumped_files/data_parts*_label.scp | sort > ${datadir}/callhome2/dumped_files/label.scp
  mkdir -p ${expdir}/callhome2_states
  awk '{print $1,"1600"}' ${datadir}/callhome2/dumped_files/feats.scp | shuf > ${expdir}/callhome2_states/speech_shape
  python -Wignore script/convert_rttm_to_seg_file.py --rttm_scp ${data_dir}/ref.rttm --seg_file ${data_dir}/org_vad.txt

fi

# Training Stage, phase 1, pretraining on simulated data with frozen encoder parameters.
# This training may cost about 1.8 days.
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: training phase 1, pretraining on simulated data"
    world_size=$gpu_num  # run on one machine
    mkdir -p ${expdir}/${model_dir}
    mkdir -p ${expdir}/${model_dir}/log
    mkdir -p /tmp/${model_dir}
    INIT_FILE=/tmp/${model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_opt=""
    if [ ! -z "${init_param}" ]; then
        init_opt="--init_param ${init_param}"
        echo ${init_opt}
    fi

    freeze_opt=""
    if [ ! -z "${freeze_param}" ]; then
        freeze_opt="--freeze_param ${freeze_param}"
        echo ${freeze_opt}
    fi

    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $gpu_devices | cut -d',' -f$[$i+1])
            python -m funasr.bin.diar_train \
                --gpu_id $gpu_id \
                --use_preprocessor false \
                --token_type char \
                --token_list $token_list \
                --train_data_path_and_name_and_type ${datadir}/${train_set}/dumped_files/feats.scp,speech,kaldi_ark \
                --train_data_path_and_name_and_type ${datadir}/${train_set}/dumped_files/profile.scp,profile,kaldi_ark \
                --train_data_path_and_name_and_type ${datadir}/${train_set}/dumped_files/label.scp,binary_labels,kaldi_ark \
                --train_shape_file ${expdir}/${train_set}_states/speech_shape \
                --valid_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/feats.scp,speech,kaldi_ark \
                --valid_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/profile.scp,profile,kaldi_ark \
                --valid_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/label.scp,binary_labels,kaldi_ark \
                --valid_shape_file ${expdir}/${valid_set}_states/speech_shape \
                --init_param ${expdir}/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch/sv.pth:encoder:encoder \
                --unused_parameters true \
                --freeze_param encoder \
                ${init_opt} \
                ${freeze_opt} \
                --ignore_init_mismatch true \
                --resume true \
                --output_dir ${expdir}/${model_dir} \
                --config $train_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${expdir}/${model_dir}/log/train.log.$i 2>&1
        } &
        done
        echo "Training log can be found at ${expdir}/${model_dir}/log/train.log.*"
        wait
fi

# evaluate for pretrained model
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11: evaluation for phase-1 model."
    for dset in ${test_sets}; do
        echo "Processing for $dset"
        exp_model_dir=${expdir}/${model_dir}
        _inference_tag="$(basename "${inference_config}" .yaml)${inference_tag}"
        _dir="${exp_model_dir}/${_inference_tag}/${inference_model}/${dset}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "WARNING: ${_dir} is already exists."
        fi
        mkdir -p "${_logdir}"
        _data="${datadir}/${dset}/dumped_files"
        key_file=${_data}/feats.scp
        num_scp_file="$(<${key_file} wc -l)"
        _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
        split_scps=
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        _opt=
        if [ ! -z "${inference_config}" ]; then
          _opt="--config ${inference_config}"
        fi
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        echo "Inference log can be found at ${_logdir}/inference.*.log"
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
            python -m funasr.bin.diar_inference_launch \
                --batch_size 1 \
                --ngpu "${_ngpu}" \
                --njob ${njob} \
                --gpuid_list ${gpuid_list} \
                --data_path_and_name_and_type "${_data}/feats.scp,speech,kaldi_ark" \
                --data_path_and_name_and_type "${_data}/profile.scp,profile,kaldi_ark" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --diar_train_config "${exp_model_dir}"/config.yaml \
                --diar_model_file "${exp_model_dir}"/"${inference_model}" \
                --output_dir "${_logdir}"/output.JOB \
                --mode sond ${_opt}
    done
fi

# Scoring for pretrained model, you may get a DER like 13.29 16.54
# 13.29: with oracle VAD, 16.54: with only SOND outputs, aka, system VAD.
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  echo "stage 12: Scoring phase-1 models"
  if [ ! -e dscore ]; then
    git clone https://github.com/nryant/dscore.git
    # add intervaltree to setup.py
  fi
  for dset in ${test_sets}; do
    echo "stage 12: Scoring for ${dset}"
    diar_exp=${expdir}/${model_dir}
    _data="${datadir}/${dset}"
    _inference_tag="$(basename "${inference_config}" .yaml)${inference_tag}"
    _dir="${diar_exp}/${_inference_tag}/${inference_model}/${dset}"
    _logdir="${_dir}/logdir"
    cat ${_logdir}/*/labels.txt | sort > ${_dir}/labels.txt

    cmd="python -Wignore script/convert_label_to_rttm.py ${_dir}/labels.txt ${datadir}/${dset}/files_for_dump/org_vad.txt ${_dir}/sys.rttm \
           --ignore_len 10 --no_pbar --smooth_size 83 --vote_prob 0.5 --n_spk 16"
    # echo ${cmd}
    eval ${cmd}
    ref=${datadir}/${dset}/files_for_dump/ref.rttm
    sys=${_dir}/sys.rttm.ref_vad
    OVAD_DER=$(python -Wignore dscore/score.py -r $ref -s $sys --collar 0.25 2>&1 | grep OVERALL | awk '{print $4}')

    ref=${datadir}/${dset}/files_for_dump/ref.rttm
    sys=${_dir}/sys.rttm.sys_vad
    SysVAD_DER=$(python -Wignore dscore/score.py -r $ref -s $sys --collar 0.25 2>&1 | grep OVERALL | awk '{print $4}')

    echo -e "${inference_model} ${OVAD_DER} ${SysVAD_DER}" | tee -a ${_dir}/results.txt
  done
fi

# Training Stage, phase 2, training on simulated data without frozen parameters.
# For V100-16G, please set batch_size to 8 in the config, and use 4 GPU to train the model with options like --gpu_devices 4,5,6,7 --gpu_num 4.
# For V100-32G, please set batch_size to 16 in the config, and use 2 GPU to train the model with options like --gpu_devices 4,5,6,7 --gpu_num 2.
# This training may cost about 3.5 days.
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "stage 13: training phase 2, training on simulated data"
    world_size=$gpu_num  # run on one machine
    mkdir -p ${expdir}/${model_dir}_phase2
    mkdir -p ${expdir}/${model_dir}_phase2/log
    mkdir -p /tmp/${model_dir}_phase2
    INIT_FILE=/tmp/${model_dir}_phase2/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_opt=""
    if [ ! -z "${init_param}" ]; then
        init_opt="--init_param ${init_param}"
        echo ${init_opt}
    fi

    freeze_opt=""
    if [ ! -z "${freeze_param}" ]; then
        freeze_opt="--freeze_param ${freeze_param}"
        echo ${freeze_opt}
    fi

    phase2_config="$(dirname "${train_config}")/$(basename "${train_config}" .yaml)_phase2.yaml"

    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $gpu_devices | cut -d',' -f$[$i+1])
            python -m funasr.bin.diar_train \
                --gpu_id $gpu_id \
                --use_preprocessor false \
                --token_type char \
                --token_list $token_list \
                --train_data_path_and_name_and_type ${datadir}/${train_set}/dumped_files/feats.scp,speech,kaldi_ark \
                --train_data_path_and_name_and_type ${datadir}/${train_set}/dumped_files/profile.scp,profile,kaldi_ark \
                --train_data_path_and_name_and_type ${datadir}/${train_set}/dumped_files/label.scp,binary_labels,kaldi_ark \
                --train_shape_file ${expdir}/${train_set}_states/speech_shape \
                --valid_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/feats.scp,speech,kaldi_ark \
                --valid_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/profile.scp,profile,kaldi_ark \
                --valid_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/label.scp,binary_labels,kaldi_ark \
                --valid_shape_file ${expdir}/${valid_set}_states/speech_shape \
                --init_param exp/${model_dir}/valid.der.ave_5best.pb \
                --unused_parameters true \
                ${init_opt} \
                ${freeze_opt} \
                --ignore_init_mismatch true \
                --resume true \
                --output_dir ${expdir}/${model_dir}_phase2 \
                --config ${phase2_config} \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${expdir}/${model_dir}_phase2/log/train.log.$i 2>&1
        } &
        done
        echo "Training log can be found at ${expdir}/${model_dir}_phase2/log/train.log.*"
        wait
fi

# evaluate for phase-2 model
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    echo "stage 14: evaluation for phase-2 model ${inference_model}."
    for dset in ${test_sets}; do
        echo "Processing for $dset"
        exp_model_dir=${expdir}/${model_dir}_phase2
        _inference_tag="$(basename "${inference_config}" .yaml)${inference_tag}"
        _dir="${exp_model_dir}/${_inference_tag}/${inference_model}/${dset}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "WARNING: ${_dir} is already exists."
        fi
        mkdir -p "${_logdir}"
        _data="${datadir}/${dset}/dumped_files"
        key_file=${_data}/feats.scp
        num_scp_file="$(<${key_file} wc -l)"
        _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
        split_scps=
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        _opt=
        if [ ! -z "${inference_config}" ]; then
          _opt="--config ${inference_config}"
        fi
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        echo "Inference log can be found at ${_logdir}/inference.*.log"
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
            python -m funasr.bin.diar_inference_launch \
                --batch_size 1 \
                --ngpu "${_ngpu}" \
                --njob ${njob} \
                --gpuid_list ${gpuid_list} \
                --data_path_and_name_and_type "${_data}/feats.scp,speech,kaldi_ark" \
                --data_path_and_name_and_type "${_data}/profile.scp,profile,kaldi_ark" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --diar_train_config "${exp_model_dir}"/config.yaml \
                --diar_model_file "${exp_model_dir}"/${inference_model} \
                --output_dir "${_logdir}"/output.JOB \
                --mode sond ${_opt}
    done
fi

# Scoring for pretrained model, you may get a DER like 11.54 15.41
# 11.54: with oracle VAD, 15.41: with only SOND outputs, aka, system VAD.
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
  echo "stage 15: Scoring phase-2 models"
  if [ ! -e dscore ]; then
    git clone https://github.com/nryant/dscore.git
    # add intervaltree to setup.py
  fi
  for dset in ${test_sets}; do
    echo "stage 15: Scoring for ${dset}"
    diar_exp=${expdir}/${model_dir}_phase2
    _data="${datadir}/${dset}"
    _inference_tag="$(basename "${inference_config}" .yaml)${inference_tag}"
    _dir="${diar_exp}/${_inference_tag}/${inference_model}/${dset}"
    _logdir="${_dir}/logdir"
    cat ${_logdir}/*/labels.txt | sort > ${_dir}/labels.txt

    cmd="python -Wignore script/convert_label_to_rttm.py ${_dir}/labels.txt ${datadir}/${dset}/files_for_dump/org_vad.txt ${_dir}/sys.rttm \
           --ignore_len 10 --no_pbar --smooth_size 83 --vote_prob 0.5 --n_spk 16"
    # echo ${cmd}
    eval ${cmd}
    ref=${datadir}/${dset}/files_for_dump/ref.rttm
    sys=${_dir}/sys.rttm.ref_vad
    OVAD_DER=$(python -Wignore dscore/score.py -r $ref -s $sys --collar 0.25 2>&1 | grep OVERALL | awk '{print $4}')

    ref=${datadir}/${dset}/files_for_dump/ref.rttm
    sys=${_dir}/sys.rttm.sys_vad
    SysVAD_DER=$(python -Wignore dscore/score.py -r $ref -s $sys --collar 0.25 2>&1 | grep OVERALL | awk '{print $4}')

    echo -e "${inference_model} ${OVAD_DER} ${SysVAD_DER}" | tee -a ${_dir}/results.txt
  done
fi


# Finetune Stage, phase 3, training on callhom1 training set
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    echo "stage 16: training phase 3, finetuing on callhome1 real data"
    world_size=$gpu_num  # run on one machine
    mkdir -p ${expdir}/${model_dir}_phase3
    mkdir -p ${expdir}/${model_dir}_phase3/log
    mkdir -p /tmp/${model_dir}_phase3
    INIT_FILE=/tmp/${model_dir}_phase3/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_opt=""
    if [ ! -z "${init_param}" ]; then
        init_opt="--init_param ${init_param}"
        echo ${init_opt}
    fi

    freeze_opt=""
    if [ ! -z "${freeze_param}" ]; then
        freeze_opt="--freeze_param ${freeze_param}"
        echo ${freeze_opt}
    fi

    phase3_config="$(dirname "${train_config}")/$(basename "${train_config}" .yaml)_phase3.yaml"

    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $gpu_devices | cut -d',' -f$[$i+1])
            python -m funasr.bin.diar_train \
                --gpu_id $gpu_id \
                --use_preprocessor false \
                --token_type char \
                --token_list $token_list \
                --train_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/feats.scp,speech,kaldi_ark \
                --train_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/profile.scp,profile,kaldi_ark \
                --train_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/label.scp,binary_labels,kaldi_ark \
                --train_shape_file ${expdir}/${valid_set}_states/speech_shape \
                --valid_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/feats.scp,speech,kaldi_ark \
                --valid_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/profile.scp,profile,kaldi_ark \
                --valid_data_path_and_name_and_type ${datadir}/${valid_set}/dumped_files/label.scp,binary_labels,kaldi_ark \
                --valid_shape_file ${expdir}/${valid_set}_states/speech_shape \
                --init_param exp/${model_dir}_phase2/valid.forward_steps.ave_5best.pb \
                --unused_parameters true \
                ${init_opt} \
                ${freeze_opt} \
                --ignore_init_mismatch true \
                --resume true \
                --output_dir ${expdir}/${model_dir}_phase3 \
                --config ${phase3_config} \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${expdir}/${model_dir}_phase3/log/train.log.$i 2>&1
        } &
        done
        echo "Training log can be found at ${expdir}/${model_dir}_phase3/log/train.log.*"
        wait
fi

# evaluate for finetuned model
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
    echo "stage 17: evaluation for finetuned model ${inference_model}."
    for dset in ${test_sets}; do
        echo "Processing for $dset"
        exp_model_dir=${expdir}/${model_dir}_phase3
        _inference_tag="$(basename "${inference_config}" .yaml)${inference_tag}"
        _dir="${exp_model_dir}/${_inference_tag}/${inference_model}/${dset}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "WARNING: ${_dir} is already exists."
        fi
        mkdir -p "${_logdir}"
        _data="${datadir}/${dset}/dumped_files"
        key_file=${_data}/feats.scp
        num_scp_file="$(<${key_file} wc -l)"
        _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
        split_scps=
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        _opt=
        if [ ! -z "${inference_config}" ]; then
          _opt="--config ${inference_config}"
        fi
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        echo "Inference log can be found at ${_logdir}/inference.*.log"
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
            python -m funasr.bin.diar_inference_launch \
                --batch_size 1 \
                --ngpu "${_ngpu}" \
                --njob ${njob} \
                --gpuid_list ${gpuid_list} \
                --data_path_and_name_and_type "${_data}/feats.scp,speech,kaldi_ark" \
                --data_path_and_name_and_type "${_data}/profile.scp,profile,kaldi_ark" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --diar_train_config "${exp_model_dir}"/config.yaml \
                --diar_model_file "${exp_model_dir}"/${inference_model} \
                --output_dir "${_logdir}"/output.JOB \
                --mode sond ${_opt}
    done
fi

# average 3 4 5 6 7 epoch
# Scoring for pretrained model, you may get a DER like
# 7.21 8.05 on callhome1
# 8.31 9.32 on callhome2
if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
  echo "stage 18: Scoring finetuned models"
  if [ ! -e dscore ]; then
    git clone https://github.com/nryant/dscore.git
    # add intervaltree to setup.py
  fi
  for dset in ${test_sets}; do
    echo "stage 18: Scoring for ${dset}"
    diar_exp=${expdir}/${model_dir}_phase3
    _data="${datadir}/${dset}"
    _inference_tag="$(basename "${inference_config}" .yaml)${inference_tag}"
    _dir="${diar_exp}/${_inference_tag}/${inference_model}/${dset}"
    _logdir="${_dir}/logdir"
    cat ${_logdir}/*/labels.txt | sort > ${_dir}/labels.txt

    cmd="python -Wignore script/convert_label_to_rttm.py ${_dir}/labels.txt ${datadir}/${dset}/files_for_dump/org_vad.txt ${_dir}/sys.rttm \
           --ignore_len 10 --no_pbar --smooth_size 83 --vote_prob 0.5 --n_spk 16"
    echo ${cmd}
    eval ${cmd}
    ref=${datadir}/${dset}/files_for_dump/ref.rttm
    sys=${_dir}/sys.rttm.ref_vad
    OVAD_DER=$(python -Wignore dscore/score.py -r $ref -s $sys --collar 0.25 2>&1 | grep OVERALL | awk '{print $4}')

    ref=${datadir}/${dset}/files_for_dump/ref.rttm
    sys=${_dir}/sys.rttm.sys_vad
    SysVAD_DER=$(python -Wignore dscore/score.py -r $ref -s $sys --collar 0.25 2>&1 | grep OVERALL | awk '{print $4}')

    echo -e "${inference_model} ${OVAD_DER} ${SysVAD_DER}" | tee -a ${_dir}/results.txt
  done
fi


if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
  for dset in ${test_sets}; do
    echo "stage 19: Evaluating phase-3 system on ${dset} set with medfilter_size=83 clustering=EEND-OLA"
    sv_exp_dir=${expdir}/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch
    diar_exp=${expdir}/${model_dir}_phase3
    _data="${datadir}/${dset}/dumped_files"
    _inference_tag="$(basename "${inference_config}" .yaml)${inference_tag}"
    _dir="${diar_exp}/${_inference_tag}/${inference_model}/${dset}"

    for iter in `seq 0 ${told_max_iter}`; do
      eval_dir=${_dir}/iter_${iter}
      if [ $iter -eq 0 ]; then
        prev_rttm=${expdir}/EEND-OLA/sys.rttm
      else
        prev_rttm=${_dir}/iter_$((${iter}-1))/sys.rttm.sys_vad
      fi
      echo "Use ${prev_rttm} as system outputs."

      echo "Iteration ${iter}, step 1: extracting non-overlap segments"
      cmd="python -Wignore script/extract_nonoverlap_segments.py ${datadir}/${dset}/wav.scp \
        $prev_rttm ${eval_dir}/nonoverlap_segs/ --min_dur 0.1 --max_spk_num 16 --no_pbar --sr 8000"
      # echo ${cmd}
      eval ${cmd}

      echo "Iteration ${iter}, step 2: make data directory"
      mkdir -p ${eval_dir}/data
      find `pwd`/${eval_dir}/nonoverlap_segs/ -iname "*.wav" | sort > ${eval_dir}/data/wav.flist
      awk -F'[/.]' '{print $(NF-1),$0}' ${eval_dir}/data/wav.flist > ${eval_dir}/data/wav.scp
      awk -F'[/.]' '{print $(NF-1),$(NF-2)}' ${eval_dir}/data/wav.flist > ${eval_dir}/data/utt2spk
      cp $prev_rttm ${eval_dir}/data/sys.rttm
      home_path=`pwd`

      echo "Iteration ${iter}, step 3: calc x-vector for each utt"
      key_file=${eval_dir}/data/wav.scp
      num_scp_file="$(<${key_file} wc -l)"
      _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
      _logdir=${eval_dir}/data/xvecs
      mkdir -p ${_logdir}
      split_scps=
      for n in $(seq "${_nj}"); do
          split_scps+=" ${_logdir}/keys.${n}.scp"
      done
      # shellcheck disable=SC2086
      utils/split_scp.pl "${key_file}" ${split_scps}

      ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/sv_inference.JOB.log \
        python -m funasr.bin.sv_inference_launch \
          --njob ${njob} \
          --batch_size 1 \
          --ngpu "${_ngpu}" \
          --gpuid_list ${gpuid_list} \
          --data_path_and_name_and_type "${key_file},speech,sound" \
          --key_file "${_logdir}"/keys.JOB.scp \
          --sv_train_config ${sv_exp_dir}/sv.yaml \
          --sv_model_file ${sv_exp_dir}/sv.pth \
          --output_dir "${_logdir}"/output.JOB
      cat ${_logdir}/output.*/xvector.scp | sort > ${eval_dir}/data/utt2xvec

      echo "Iteration ${iter}, step 4: dump x-vector record"
      awk '{print $1}' ${_data}/feats.scp > ${eval_dir}/data/idx
      python script/dump_speaker_profiles.py --dir ${eval_dir}/data \
        --out ${eval_dir}/global_n16 --n_spk 16 --no_pbar --emb_type global
      spk_profile=${eval_dir}/global_n16_parts00_xvec.scp

      echo "Iteration ${iter}, step 5: perform NN diarization"
      _logdir=${eval_dir}/diar
      mkdir -p ${_logdir}
      key_file=${_data}/feats.scp
      num_scp_file="$(<${key_file} wc -l)"
      _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
      split_scps=
      for n in $(seq "${_nj}"); do
          split_scps+=" ${_logdir}/keys.${n}.scp"
      done
      _opt=
      if [ ! -z "${inference_config}" ]; then
        _opt="--config ${inference_config}"
      fi
      # shellcheck disable=SC2086
      utils/split_scp.pl "${key_file}" ${split_scps}

      echo "Inference log can be found at ${_logdir}/inference.*.log"
      ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
          python -m funasr.bin.diar_inference_launch \
              --batch_size 1 \
              --ngpu "${_ngpu}" \
              --njob ${njob} \
              --gpuid_list ${gpuid_list} \
              --data_path_and_name_and_type "${_data}/feats.scp,speech,kaldi_ark" \
              --data_path_and_name_and_type "${spk_profile},profile,kaldi_ark" \
              --key_file "${_logdir}"/keys.JOB.scp \
              --diar_train_config ${diar_exp}/config.yaml \
              --diar_model_file ${diar_exp}/${inference_model} \
              --output_dir "${_logdir}"/output.JOB \
              --mode sond ${_opt}

      echo "Iteration ${iter}, step 6: calc diarization results"
      cat ${_logdir}/output.*/labels.txt | sort > ${eval_dir}/labels.txt

      cmd="python -Wignore script/convert_label_to_rttm.py ${eval_dir}/labels.txt ${datadir}/${dset}/files_for_dump/org_vad.txt ${eval_dir}/sys.rttm \
             --ignore_len 10 --no_pbar --smooth_size 83 --vote_prob 0.5 --n_spk 16"
      # echo ${cmd}
      eval ${cmd}
      ref=${datadir}/${dset}/files_for_dump/ref.rttm
      sys=${eval_dir}/sys.rttm.ref_vad
      OVAD_DER=$(python -Wignore dscore/score.py -r $ref -s $sys --collar 0.25 2>&1 | grep OVERALL | awk '{print $4}')

      ref=${datadir}/${dset}/files_for_dump/ref.rttm
      sys=${eval_dir}/sys.rttm.sys_vad
      SysVAD_DER=$(python -Wignore dscore/score.py -r $ref -s $sys --collar 0.25 2>&1 | grep OVERALL | awk '{print $4}')

      echo -e "${inference_model}/iter_${iter} ${OVAD_DER} ${SysVAD_DER}" | tee -a ${eval_dir}/results.txt
    done

    echo "Done."
  done
fi
