from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

multilingual_wavs=[
    "https://www.modelscope.cn/api/v1/models/iic/speech_whisper-large_lid_multilingual_pytorch/repo?Revision=master&FilePath=examples/example_zh-CN.mp3",
    "https://www.modelscope.cn/api/v1/models/iic/speech_whisper-large_lid_multilingual_pytorch/repo?Revision=master&FilePath=examples/example_en.mp3",
    "https://www.modelscope.cn/api/v1/models/iic/speech_whisper-large_lid_multilingual_pytorch/repo?Revision=master&FilePath=examples/example_ja.mp3",
    "https://www.modelscope.cn/api/v1/models/iic/speech_whisper-large_lid_multilingual_pytorch/repo?Revision=master&FilePath=examples/example_ko.mp3",
]

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/speech_whisper-large_lid_multilingual_pytorch', model_revision="v0.0.2")

for wav in multilingual_wavs:
    rec_result = inference_pipeline(input=wav)
    print(rec_result)