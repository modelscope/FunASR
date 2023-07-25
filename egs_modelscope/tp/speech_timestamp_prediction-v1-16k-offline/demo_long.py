# if you want to use ASR model besides paraformer-bicif (like contextual paraformer)
# to get ASR results for long audio as well as timestamp prediction results, 
# try this demo
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import librosa
import soundfile as sf

param_dict = dict()
param_dict['hotword'] = "你的热词"

test_wav = 'YOUR_LONG_WAV.wav'
output_dir = './tmp'
os.system("mkdir -p {}".format(output_dir))

vad_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision=None,
)
asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    output_dir=output_dir)
tp_pipeline = pipeline(
    task=Tasks.speech_timestamp,
    model='damo/speech_timestamp_prediction-v1-16k-offline',
    output_dir=output_dir)

vad_res = vad_pipeline(audio_in=test_wav)
timestamps = vad_res['text']

samples = librosa.load(test_wav, sr=16000)[0]
wavseg_scp = "{}/wav.scp".format(output_dir)

with open(wavseg_scp, 'w') as fout:
    for i, timestamp in enumerate(timestamps):
        start = int(timestamp[0]/1000*16000)
        end = int(timestamp[1]/1000*16000)
        uttid = "wav_{}_{} ".format(start, end)
        wavpath = '{}/wavseg_{}.wav'.format(output_dir, i)
        _samples = samples[start:end]
        sf.write(wavpath, _samples, 16000)
        fout.write("{} {}\n".format(uttid, wavpath))
print("Wav segment done: {}".format(wavseg_scp))

asr_res = '{}/1best_recog/text'.format(output_dir)
tp_res = '{}/timestamp_prediction/tp_sync'.format(output_dir)
rec_result_asr = asr_pipeline(audio_in=wavseg_scp)
rec_result_tp = tp_pipeline(audio_in=wavseg_scp, text_in=asr_res)
print("Find your ASR results in {}, and timestamp prediction results in {}.".format(asr_res, tp_res))