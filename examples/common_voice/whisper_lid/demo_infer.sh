# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)



python -m funasr.bin.inference \
--config-path="/nfs/wangjiaming.wjm/Funasr_Results/Whisper_LargeV3-LID/CommonVoiceFleursVoxLingual107/2m-4gpu/WhisperLID" \
--config-name="config.yaml" \
++init_param="/nfs/wangjiaming.wjm/Funasr_Results/Whisper_LargeV3-LID/CommonVoiceFleursVoxLingual107/2m-4gpu/WhisperLID/model.pth" \
++tokenizer="CharTokenizer" \
++tokenizer_conf.token_list="/nfs/wangjiaming.wjm/Funasr_data/Whisper_LID/common_voice_fleurs_voxlingual107/tokens.txt" \
++input="/nfs/wangjiaming.wjm/Funasr_data/multilingual/common_voice/cv-corpus-15.0-2023-09-08/zh-CN/test_mp3/wavelist_local.scp" \
++batch_size=1 \
++output_dir="./outputs/debug" \
++device="cuda:0"

