from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.speech_timestamp,
    model='damo/speech_timestamp_prediction-v1-16k-offline',
    output_dir=None)

rec_result = inference_pipeline(
    audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_timestamps.wav',
    text_in='一 个 东 太 平 洋 国 家 为 什 么 跑 到 西 太 平 洋 来 了 呢',)
print(rec_result)