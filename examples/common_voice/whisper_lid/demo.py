from funasr import AutoModel

multilingual_wavs = [
    "example_zh-CN.mp3",
    "example_en.mp3",
    "example_ja.mp3",
    "example_ko.mp3",
]

model = AutoModel(model="iic/speech_whisper-large_lid_multilingual_pytorch", model_revision="v0.0.2")
for wav_id in multilingual_wavs:
    wav_file = f"{model.model_path}/examples/{wav_id}"
    res = model.generate(input=wav_file, data_type="sound")
    print("detect sample {}: {}".format(wav_id, res))