import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input = 'https://modelscope.cn/api/v1/models/damo/speech_separation_mossformer_8k_pytorch/repo?Revision=master&FilePath=examples/mix_speech1.wav'
separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_separation_mossformer_8k_pytorch',
   output_dir='./',
   model_revision='v1.0.2')
result = separation(audio_in=input)
for i, signal in enumerate(result):
    save_file = f'output_spk_{i+1}.wav'
    sf.write(save_file, signal, 8000)
