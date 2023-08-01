#!/usr/bin/env bash

. ./path.sh || exit 1;

# This recipe aims at reimplement the results of SOND on Callhome corpus which is represented in
# [1] TOLD: A Novel Two-stage Overlap-aware Framework for Speaker Diarization, ICASSP 2023
# You can also use it on other dataset such AliMeeting to reproduce the results in
# [2] Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis, EMNLP 2022
# We recommend you run this script stage by stage.

# environment configuration
if [ ! -e utils ]; then
  ln -s ../../../aishell/transformer/utils ./utils
fi

# machines configuration
gpu_devices="0,1,2,3"
gpu_num=4
count=1

# general configuration
stage=1
stop_stage=1
# number of jobs for data process
nj=16
sr=8000

# dataset related
data_root=

# experiment configuration
lang=en
feats_type=fbank
datadir=data
dumpdir=dump
expdir=exp
train_cmd=utils/run.pl

# training related
tag=""
train_set=callhome1
valid_set=callhome1
train_config=conf/EAND_ResNet34_SAN_L4N512_None_FFN_FSMN_L6N512_bce_dia_loss_01_phase3.yaml
token_list=${datadir}/token_list/powerset_label_n16k4.txt
init_param=
freeze_param=

# inference related
inference_model=valid.der.ave_5best.pth
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

# Download required resources
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: Download required resources."
  wget told_finetune_resources.zip
fi

# Finetune model on callhome1
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Finetune pretrained model on callhome1."
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
          diar_train.py \
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
              --init_param exp/pretrained_models/phase2.pth \
              --unused_parameters true \
              ${init_opt} \
              ${freeze_opt} \
              --ignore_init_mismatch true \
              --resume true \
              --output_dir ${expdir}/${model_dir} \
              --config ${train_config} \
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


# evaluate for finetuned model
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: evaluation for finetuned model ${inference_model}."
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
                --diar_model_file "${exp_model_dir}"/${inference_model} \
                --output_dir "${_logdir}"/output.JOB \
                --mode sond ${_opt}
    done
fi

# Scoring for finetuned model, you may get a DER like
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Scoring finetuned models"
  if [ ! -e dscore ]; then
    git clone https://github.com/nryant/dscore.git
    # add intervaltree to setup.py
  fi
  for dset in ${test_sets}; do
    echo "stage 3: Scoring for ${dset}"
    diar_exp=${expdir}/${model_dir}
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

# In this stage, we need the raw waveform files of Callhome corpus.
# Due to the data license, we can't provide them, please get them additionally.
# And convert the sph files to wav files (use scripts/dump_pipe_wav.py).
# Then find the wav files to construct wav.scp and put it at data/callhome2/wav.scp.
# After iteratively perform SOAP, you will get DER results like:
# iters| oracle_vad  |  system_vad
# iter_0:   9.68      |     10.51
# iter_1:   9.26      |     10.14  (reported in the paper)
# iter_2:   9.18      |     10.08
# iter_3:   9.24      |     10.15
# iter_4:   9.27      |     10.17
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  for dset in ${test_sets}; do
    echo "stage 4: Evaluating finetuned system on ${dset} set with medfilter_size=83 clustering=EEND-OLA"
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
