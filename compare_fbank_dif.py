import torch
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
import kaldi_native_fbank as knf
import soundfile as sf
import numpy as np
# 读取音频文件
# audio_file = '/home/wzp/project/FunASR/asr_example_zh.wav'
audio_file = '/home/wzp/project/FunASR/speaker1_a_cn_16k.wav'
# 读取音频文件
sampling_rate = 16000
mel_bins = 80
frame_length = 25  # ms
frame_shift = 10  # ms
dither = 0
preemphasis_coefficient = 0.97
window_type = 'povey'

waveform, _ = torchaudio.load(audio_file)

# 使用 torchaudio.compliance.kaldi 提取 fbank 特征
kaldi_fbank = Kaldi.fbank(
    waveform,
    num_mel_bins=mel_bins,
    frame_length=frame_length,
    frame_shift=frame_shift,
    dither=dither,
    # preemph_coef=preemphasis_coefficient,
    window_type=window_type,
    sample_frequency=sampling_rate
)
kaldi_fbank = kaldi_fbank.numpy()
print("kaldi_fbank,shape", kaldi_fbank.shape)

# 使用 kaldi_native_fbank 提取 fbank 特征
samples, _ = sf.read(audio_file)
opts = knf.FbankOptions()
opts.frame_opts.dither = dither
opts.frame_opts.preemph_coeff = preemphasis_coefficient
opts.frame_opts.samp_freq = sampling_rate
# opts.frame_opts.frame_shift = frame_shift / 1000.0  # 秒
# opts.frame_opts.frame_length = frame_length / 1000.0  # 秒
opts.frame_opts.window_type = window_type
opts.mel_opts.num_bins = mel_bins
opts.mel_opts.debug_mel = False
fbank = knf.OnlineFbank(opts)
fbank.accept_waveform(sampling_rate, samples.tolist())

print("fbank.num_frames_ready", fbank.num_frames_ready)

native_fbank=[]
for i in range(fbank.num_frames_ready):
    f2 = fbank.get_frame(i)
    native_fbank.append(f2)
native_fbank=np.array(native_fbank)
print("native_fbank shape",native_fbank.shape)
print(native_fbank)
print("====")
print(kaldi_fbank)
difference = native_fbank-kaldi_fbank
difference_l2_norm = np.linalg.norm(difference, ord=2)
print("L2 norm of the difference between the two fbank features:", difference_l2_norm)

