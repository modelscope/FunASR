"""
Author: Speech Lab, Alibaba Group, China
SOND: Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis
https://arxiv.org/abs/2211.10243
"""

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# initialize the pipeline for inference
# when using the raw waveform files to inference, please use the config file `sond.yaml`
# and set mode to `sond_demo`
inference_diar_pipline = pipeline(
    mode="sond_demo",
    num_workers=0,
    task=Tasks.speaker_diarization,
    diar_model_config="sond.yaml",
    model='damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch',
    sv_model="damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch",
    sv_model_revision="v1.2.2",
)

# use audio_list as the input, where the first one is the record to be detected
# and the following files are enrollments for different speakers
audio_list = [
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/record.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk1.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk2.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk3.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk4.wav",
]

results = inference_diar_pipline(audio_in=audio_list)
print(results)
