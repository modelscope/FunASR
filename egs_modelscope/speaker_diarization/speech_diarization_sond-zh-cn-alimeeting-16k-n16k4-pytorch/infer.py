from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 初始化推理 pipeline
# 当以原始音频作为输入时使用配置文件 sond.yaml，并设置 mode 为sond_demo
inference_diar_pipline = pipeline(
    mode="sond_demo",
    num_workers=0,
    task=Tasks.speaker_diarization,
    diar_model_config="sond.yaml",
    model='damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch',
    sv_model="damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch",
    sv_model_revision="master",
)

# 以 audio_list 作为输入，其中第一个音频为待检测语音，后面的音频为不同说话人的声纹注册语音
audio_list = [
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/record.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk1.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk2.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk3.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk4.wav",
]

results = inference_diar_pipline(audio_in=audio_list)
print(results)
