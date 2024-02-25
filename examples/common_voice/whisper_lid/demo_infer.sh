# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)



python -m funasr.bin.inference \
--config-path="/nfs/wangjiaming.wjm/Funasr_Results/Whisper_LargeV3-LID/CommonVoiceFleursVoxLingual107/2m-4gpu/WhisperLID" \
--config-name="config.yaml" \
++init_param="/nfs/wangjiaming.wjm/Funasr_Results/Whisper_LargeV3-LID/CommonVoiceFleursVoxLingual107/2m-4gpu/WhisperLI/model.pth" \
++tokenizer_conf.token_list="/nfs/wangjiaming.wjm/Funasr_data/Whisper_LID/common_voice_fleurs_voxlingual107/tokens.txt" \
++input="/nfs/yangyexin.yyx/data/multilingual/common_voice/cv-corpus-15.0-2023-09-08-zh-CN/cv-corpus-15.0-2023-09-08/zh-CN/clips/common_voice_zh-CN_22114629.mp3" \
++output_dir="./outputs/debug" \
++device="cuda:0" \

