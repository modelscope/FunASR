# Introduction

Kaldi-compatible online fbank feature extractor without external dependencies.

Tested on the following architectures and operating systems:

  - Linux
  - macOS
  - Windows
  - Android
  - x86
  - arm
  - aarch64

# Usage

See the following CMake-based speech recognition (i.e., text-to-speech) projects
for its usage:

- <https://github.com/k2-fsa/sherpa-ncnn>
  - Specifically, please have a look at <https://github.com/k2-fsa/sherpa-ncnn/blob/master/sherpa-ncnn/csrc/features.h>
- <https://github.com/k2-fsa/sherpa-onnx>

They use `kaldi-native-fbank` to compute fbank features for **real-time**
speech recognition.

# Python APIs

First, please install `kaldi-native-fbank` by

```bash
git clone https://github.com/csukuangfj/kaldi-native-fbank
cd kaldi-native-fbank
python3 setup.py install
```

or use

```bash
pip install kaldi-native-fbank
```

To check that you have installed `kaldi-native-fbank` successfully, please use

```
python3 -c "import kaldi_native_fbank; print(kaldi_native_fbank.__version__)"
```

which should print the version you have installed.

Please refer to
<https://github.com/csukuangfj/kaldi-native-fbank/blob/master/kaldi-native-fbank/python/tests/test_online_fbank.py>
for usages.

For easier reference, we post the above file below:

```python3
#!/usr/bin/env python3

import sys

try:
    import kaldifeat
except:
    print("Please install kaldifeat first")
    sys.exit(0)

import kaldi_native_fbank as knf
import torch


def main():
    sampling_rate = 16000
    samples = torch.randn(16000 * 10)

    opts = kaldifeat.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False

    online_fbank = kaldifeat.OnlineFbank(opts)

    online_fbank.accept_waveform(sampling_rate, samples)

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sampling_rate, samples.tolist())

    assert online_fbank.num_frames_ready == fbank.num_frames_ready
    for i in range(fbank.num_frames_ready):
        f1 = online_fbank.get_frame(i)
        f2 = torch.from_numpy(fbank.get_frame(i))
        assert torch.allclose(f1, f2, atol=1e-3), (i, (f1 - f2).abs().max())


if __name__ == "__main__":
    torch.manual_seed(20220825)
    main()
    print("success")
```
