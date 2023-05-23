"""
Author: Speech Lab, Alibaba Group, China
TOLD: A Novel Two-Stage Overlap-Aware Framework for Speaker Diarization
https://arxiv.org/abs/2303.05397
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
    model='damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch',
    sv_model="damo/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch",
    sv_model_revision="master",
)

# use audio_list as the input, where the first one is the record to be detected
# and the following files are enrollments for different speakers
audio_list = [
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/record.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/spk_A.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/spk_B.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/spk_B1.wav"
]

results = inference_diar_pipline(audio_in=audio_list)
print(results)
