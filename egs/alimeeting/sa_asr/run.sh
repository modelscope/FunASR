#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="6,7"
gpu_num=2
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=8
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
feats_dir="data" #feature output dictionary
exp_dir="exp"
lang=zh
token_type=char
type=sound
scp=wav.scp
speed_perturb="1.0"
min_wav_duration=0.1
max_wav_duration=20
profile_modes="cluster oracle"
stage=7
stop_stage=7

# feature configuration
feats_dim=80
nj=32

# data
raw_data=
data_url=

# exp tag
tag=""

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=Train_Ali_far
valid_set=Eval_Ali_far
test_sets="Test_Ali_far Eval_Ali_far"
test_2023="Test_2023_Ali_far_release"

asr_config=conf/train_asr_conformer.yaml
sa_asr_config=conf/train_sa_asr_conformer.yaml
asr_model_dir="baseline_$(basename "${asr_config}" .yaml)_${lang}_${token_type}_${tag}"
sa_asr_model_dir="baseline_$(basename "${sa_asr_config}" .yaml)_${lang}_${token_type}_${tag}"
inference_config=conf/decode_asr_rnn.yaml
inference_sa_asr_model=valid.acc_spk.ave.pb

# you can set gpu num for decoding here
gpuid_list=$CUDA_VISIBLE_DEVICES  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # Data preparation
    ./local/alimeeting_data_prep.sh --tgt Test --min_wav_duration $min_wav_duration --max_wav_duration $max_wav_duration
    ./local/alimeeting_data_prep.sh --tgt Eval --min_wav_duration $min_wav_duration --max_wav_duration $max_wav_duration
    ./local/alimeeting_data_prep.sh --tgt Train --min_wav_duration $min_wav_duration --max_wav_duration $max_wav_duration
    remove long/short data
    for x in ${train_set} ${valid_set} ${test_sets}; do
        cp -r ${feats_dir}/org/${x} ${feats_dir}/${x}
        rm ${feats_dir}/"${x}"/wav.scp ${feats_dir}/"${x}"/segments
        local/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format wav --segments ${feats_dir}/org/${x}/segments \
            "${feats_dir}/org/${x}/${scp}" "${feats_dir}/${x}"
        _min_length=$(python3 -c "print(int(${min_wav_duration} * 16000))")
        _max_length=$(python3 -c "print(int(${max_wav_duration} * 16000))")
        <"${feats_dir}/${x}/utt2num_samples" \
        awk '{if($2 > '$_min_length' && $2 < '$_max_length')print $0;}' \
            >"${feats_dir}/${x}/utt2num_samples_rmls"
        mv ${feats_dir}/${x}/utt2num_samples_rmls ${feats_dir}/${x}/utt2num_samples
        <"${feats_dir}/${x}/wav.scp" \
            utils/filter_scp.pl "${feats_dir}/${x}/utt2num_samples"  \
            >"${feats_dir}/${x}/wav.scp_rmls"
        mv ${feats_dir}/${x}/wav.scp_rmls ${feats_dir}/${x}/wav.scp
        <"${feats_dir}/${x}/text" \
            awk '{ if( NF != 1 ) print $0; }' >"${feats_dir}/${x}/text_rmblank"
        mv ${feats_dir}/${x}/text_rmblank ${feats_dir}/${x}/text
        local/fix_${feats_dir}_dir.sh "${feats_dir}/${x}"
        <"${feats_dir}/${x}/utt2spk_all_fifo" \
            utils/filter_scp.pl "${feats_dir}/${x}/text"  \
            >"${feats_dir}/${x}/utt2spk_all_fifo_rmls"
        mv "${feats_dir}/${x}/utt2spk_all_fifo_rmls" "${feats_dir}/${x}/utt2spk_all_fifo"
    done
    for x in ${test_2023}; do
        local/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format wav --segments ${feats_dir}/org/${x}/segments \
            "${feats_dir}/org/${x}/${scp}" "${feats_dir}/${x}"
        cut -d " " -f1 ${feats_dir}/${x}/wav.scp > ${feats_dir}/${x}/uttid
        paste -d " " ${feats_dir}/${x}/uttid ${feats_dir}/${x}/uttid > ${feats_dir}/${x}/utt2spk
        cp ${feats_dir}/${x}/utt2spk ${feats_dir}/${x}/spk2utt
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Speaker profile and CMVN Generation"
    
    mkdir -p "profile_log"
    for x in "${train_set}" "${valid_set}" "${test_sets}"; do
        # generate text_id spk2id
        python local/process_sot_fifo_textchar2spk.py --path ${feats_dir}/${x}
        echo "Successfully generate ${feats_dir}/${x}/text_id ${feats_dir}/${x}/spk2id"
        # generate text_id_train for sot
        python local/process_text_id.py ${feats_dir}/${x}
        echo "Successfully generate ${feats_dir}/${x}/text_id_train"
        # generate oracle_embedding from single-speaker audio segment
        echo "oracle_embedding is being generated in the background, and the log is profile_log/gen_oracle_embedding_${x}.log"
        python local/gen_oracle_embedding.py "${feats_dir}/${x}" "data/org/${x}_single_speaker" &> "profile_log/gen_oracle_embedding_${x}.log"
        echo "Successfully generate oracle embedding for ${x} (${feats_dir}/${x}/oracle_embedding.scp)"
        # generate oracle_profile and cluster_profile from oracle_embedding and cluster_embedding (padding the speaker during training)
        if [ "${x}" = "${train_set}" ]; then
            python local/gen_oracle_profile_padding.py ${feats_dir}/${x}
            echo "Successfully generate oracle profile for ${x} (${feats_dir}/${x}/oracle_profile_padding.scp)"
        else
            python local/gen_oracle_profile_nopadding.py ${feats_dir}/${x}
            echo "Successfully generate oracle profile for ${x} (${feats_dir}/${x}/oracle_profile_nopadding.scp)"
        fi
        # generate cluster_profile with spectral-cluster directly (for infering and without oracle information)
        if [ "${x}" = "${valid_set}" ] || [ "${x}" = "${test_sets}" ]; then
            echo "cluster_profile is being generated in the background, and the log is profile_log/gen_cluster_profile_infer_${x}.log"
            python local/gen_cluster_profile_infer.py "${feats_dir}/${x}" "${feats_dir}/org/${x}" 0.996 0.815 &> "profile_log/gen_cluster_profile_infer_${x}.log"
            echo "Successfully generate cluster profile for ${x} (${feats_dir}/${x}/cluster_profile_infer.scp)"
        fi
        # compute CMVN
        if [ "${x}" = "${train_set}" ]; then
            local/compute_cmvn.sh --cmd "$train_cmd" --nj $nj --fbankdir ${feats_dir}/${train_set} --feats_dim ${feats_dim} --config_file "$asr_config" --scale 1.0
        fi
    done

    for x in "${test_2023}"; do
        # generate cluster_profile with spectral-cluster directly (for infering and without oracle information)
        python local/gen_cluster_profile_infer.py "${feats_dir}/${x}" "${feats_dir}/org/${x}" 0.996 0.815 &> "profile_log/gen_cluster_profile_infer_${x}.log"
        echo "Successfully generate cluster profile for ${x} (${feats_dir}/${x}/cluster_profile_infer.scp)"
    done
fi

token_list=${feats_dir}/${lang}_token_list/char/tokens.txt
echo "dictionary: ${token_list}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary Preparation"
    mkdir -p ${feats_dir}/${lang}_token_list/char/

    echo "make a dictionary"
    echo "<blank>" > ${token_list}
    echo "<s>" >> ${token_list}
    echo "</s>" >> ${token_list}
    utils/text2token.py -s 1 -n 1 --space "" ${feats_dir}/$train_set/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0}' >> ${token_list}
    echo "<unk>" >> ${token_list}
fi

# LM Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Training"
fi

# ASR Training Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: ASR Training"
    asr_exp=${exp_dir}/${asr_model_dir}
    mkdir -p ${asr_exp}
    mkdir -p ${asr_exp}/log
    INIT_FILE=${asr_exp}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi 
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $ngpu; ++i)); do
        {
            # i=0
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
            train.py \
                --task_name asr \
                --model asr \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --split_with_space false \
                --token_type char \
                --token_list $token_list \
                --data_dir ${feats_dir} \
                --train_set ${train_set} \
                --valid_set ${valid_set} \
                --data_file_names "wav.scp,text" \
                --cmvn_file ${feats_dir}/${train_set}/cmvn/cmvn.mvn \
                --speed_perturb ${speed_perturb} \
                --resume true \
                --output_dir ${exp_dir}/${asr_model_dir} \
                --config $asr_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/${asr_model_dir}/log/train.log.$i 2>&1
        } &
    done
    wait

fi



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "SA-ASR training"
    asr_exp=${exp_dir}/${asr_model_dir}
    sa_asr_exp=${exp_dir}/${sa_asr_model_dir}
    mkdir -p ${sa_asr_exp}
    mkdir -p ${sa_asr_exp}/log
    INIT_FILE=${sa_asr_exp}/ddp_init
    if [ ! -L ${feats_dir}/${train_set}/profile.scp ]; then
        ln -sr ${feats_dir}/${train_set}/oracle_profile_padding.scp ${feats_dir}/${train_set}/profile.scp
        ln -sr ${feats_dir}/${valid_set}/oracle_profile_nopadding.scp ${feats_dir}/${valid_set}/profile.scp
    fi
    
    if [ ! -f "${exp_dir}/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.pth" ]; then
        # download xvector extractor model file
        python local/download_xvector_model.py ${exp_dir}
        echo "Successfully download the pretrained xvector extractor to exp/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.pth"
    fi
    
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi 
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $ngpu; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
	        train.py \
                --task_name asr \
                --model sa_asr \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --split_with_space false \
                --unused_parameters true \
                --token_type char \
                --resume true \
                --token_list $token_list \
                --data_dir ${feats_dir} \
                --train_set ${train_set} \
                --valid_set ${valid_set} \
                --data_file_names "wav.scp,text,profile.scp,text_id_train" \
                --cmvn_file ${feats_dir}/${train_set}/cmvn/cmvn.mvn \
                --speed_perturb ${speed_perturb} \
                --init_param "${asr_exp}/valid.acc.ave.pb:encoder:asr_encoder"   \
                --init_param "${asr_exp}/valid.acc.ave.pb:ctc:ctc"   \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.embed:decoder.embed" \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.output_layer:decoder.asr_output_layer" \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.0.self_attn:decoder.decoder1.self_attn" \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.0.src_attn:decoder.decoder3.src_attn" \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.0.feed_forward:decoder.decoder3.feed_forward" \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.1:decoder.decoder4.0" \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.2:decoder.decoder4.1" \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.3:decoder.decoder4.2" \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.4:decoder.decoder4.3" \
                --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.5:decoder.decoder4.4" \
                --init_param "exp/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.pth:encoder:spk_encoder"   \
                --init_param "exp/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.pth:decoder:spk_encoder:decoder.output_dense"   \
                --output_dir ${exp_dir}/${sa_asr_model_dir} \
                --config $sa_asr_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/${sa_asr_model_dir}/log/train.log.$i 2>&1
        } &
    done
    wait
fi
                

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Inference test sets"
    for x in ${test_sets}; do
        for profile_mode in ${profile_modes}; do
            echo "decoding ${x} with ${profile_mode} profile"
            sa_asr_exp=${exp_dir}/${sa_asr_model_dir}
            inference_tag="$(basename "${inference_config}" .yaml)"
            _dir="${sa_asr_exp}/${inference_tag}_${profile_mode}/${inference_sa_asr_model}/${x}"
            _logdir="${_dir}/logdir"
            if [ -d ${_dir} ]; then
                echo "${_dir} is already exists. if you want to decode again, please delete this dir first."
                exit 0
            fi
            mkdir -p "${_logdir}"
            _data="${feats_dir}/${x}"
            key_file=${_data}/${scp}
            num_scp_file="$(<${key_file} wc -l)"
            _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
            split_scps=
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}
            _opts=
            if [ -n "${inference_config}" ]; then
                _opts+="--config ${inference_config} "
            fi
            if [ $profile_mode = "oracle" ]; then
                profile_scp=${profile_mode}_profile_nopadding.scp
            else
                profile_scp=${profile_mode}_profile_infer.scp
            fi
            ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
                python -m funasr.bin.asr_inference_launch \
                    --batch_size 1 \
                    --mc True \
                    --ngpu "${_ngpu}" \
                    --njob ${njob} \
                    --nbest 1 \
                    --gpuid_list ${gpuid_list} \
                    --allow_variable_data_keys true \
                    --cmvn_file ${feats_dir}/${train_set}/cmvn/cmvn.mvn \
                    --data_path_and_name_and_type "${_data}/${scp},speech,${type}" \
                    --data_path_and_name_and_type "${_data}/$profile_scp,profile,npy" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --asr_train_config "${sa_asr_exp}"/config.yaml \
                    --asr_model_file "${sa_asr_exp}"/"${inference_sa_asr_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    --mode sa_asr \
                    ${_opts}

            for f in token token_int score text text_id; do
                if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/1best_recog/${f}"
                    done | sort -k1 >"${_dir}/${f}"
                fi
            done
            sed 's/\$//g' ${_data}/text > ${_data}/text_nosrc
            sed 's/\$//g' ${_dir}/text > ${_dir}/text_nosrc
            python utils/proce_text.py ${_data}/text_nosrc ${_data}/text.proc
            python utils/proce_text.py ${_dir}/text_nosrc ${_dir}/text.proc

            python utils/compute_wer.py ${_data}/text.proc ${_dir}/text.proc ${_dir}/text.cer
            tail -n 3 ${_dir}/text.cer > ${_dir}/text.cer.txt
            cat ${_dir}/text.cer.txt

            python local/process_text_spk_merge.py ${_dir}
            python local/process_text_spk_merge.py ${_data}
            
            python local/compute_cpcer.py ${_data}/text_spk_merge ${_dir}/text_spk_merge ${_dir}/text.cpcer
            tail -n 1 ${_dir}/text.cpcer > ${_dir}/text.cpcer.txt
            cat ${_dir}/text.cpcer.txt
        done
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Inference test 2023"
    for x in ${test_2023}; do
        sa_asr_exp=${exp_dir}/${sa_asr_model_dir}
        inference_tag="$(basename "${inference_config}" .yaml)"
        _dir="${sa_asr_exp}/${inference_tag}_cluster/${inference_sa_asr_model}/${x}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "${_dir} is already exists. if you want to decode again, please delete this dir first."
            exit 0
        fi
        mkdir -p "${_logdir}"
        _data="${feats_dir}/${x}"
        key_file=${_data}/${scp}
        num_scp_file="$(<${key_file} wc -l)"
        _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
        split_scps=
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}
        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
            python -m funasr.bin.asr_inference_launch \
                --batch_size 1 \
                --mc True \
                --ngpu "${_ngpu}" \
                --njob ${njob} \
                --nbest 1 \
                --gpuid_list ${gpuid_list} \
                --allow_variable_data_keys true \
                --data_path_and_name_and_type "${_data}/${scp},speech,${type}" \
                --data_path_and_name_and_type "${_data}/cluster_profile_infer.scp,profile,npy" \
                --cmvn_file ${feats_dir}/${train_set}/cmvn/cmvn.mvn \
                --key_file "${_logdir}"/keys.JOB.scp \
                --asr_train_config "${sa_asr_exp}"/config.yaml \
                --asr_model_file "${sa_asr_exp}"/"${inference_sa_asr_model}" \
                --output_dir "${_logdir}"/output.JOB \
                --mode sa_asr \
                ${_opts}

        for f in token token_int score text text_id; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done

        python local/process_text_spk_merge.py ${_dir}

    done
fi


