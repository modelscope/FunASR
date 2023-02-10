#!/bin/bash

. ./path.sh || exit 1;

stage=0
stop_stage=2

. utils/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Downloading AliMeeting test set data..."
  wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/alimeeting_test_data_for_sond.tar.gz
  echo "Done. Extracting data..."
  tar zxf alimeeting_test_data_for_sond.tar.gz
  echo "Done."

  echo "Downloading Pre-trained model..."
  git clone https://www.modelscope.cn/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch.git
  git clone https://www.modelscope.cn/damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch.git
  ln -s speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.pth ./sv.pth
  cp speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.yaml ./sv.yaml
  ln -s speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch/sond.pth ./sond.pth
  cp speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch/sond_fbank.yaml ./sond_fbank.yaml
  cp speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch/sond.yaml ./sond.yaml
  echo "Done."

  echo "Downloading dscore for scoring..."
  git clone https://github.com/nryant/dscore.git
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Calculating diarization results..."
  python infer_alimeeting_test.py sond_fbank.yaml sond.pth outputs
  python local/convert_label_to_rttm.py \
    outputs/labels.txt \
    data/test_rmsil/raw_rmsil_map.scp \
    outputs/prediction_sm_83.rttm \
    --ignore_len 10 --no_pbar --smooth_size 83 \
    --vote_prob 0.5 --n_spk 16
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Scoring..."
  python dscore/score.py \
    -r data/test_rmsil/test_org.crttm \
    -s outputs/prediction_sm_83.rttm \
    --collar 0.25
fi
