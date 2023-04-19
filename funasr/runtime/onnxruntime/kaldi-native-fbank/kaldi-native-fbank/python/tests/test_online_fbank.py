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
