/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

#include "feat_computer.h"

FeatComputer::FeatComputer()
{
    fbank_opts.use_energy = false;
    fbank_opts.mel_opts.low_freq = 20.0;
    fbank_opts.mel_opts.high_freq = 0;
    fbank_opts.mel_opts.num_bins = 80;
    fbank_opts.frame_opts.samp_freq = 16000;
    fbank_opts.frame_opts.frame_shift_ms = 10.0;
    fbank_opts.frame_opts.frame_length_ms = 25.0;
    fbank_opts.frame_opts.preemph_coeff = 0.97;
    fbank_opts.frame_opts.window_type = "hamming";
    fbank_opts.frame_opts.remove_dc_offset = true;
    fbank_opts.frame_opts.dither = 1.0;

    fbank_computer = new kaldi::Fbank(fbank_opts);
}

void FeatComputer::compute_signals(kaldi::SubVector<kaldi::BaseFloat> signals,
                                   kaldi::Matrix<kaldi::BaseFloat> *output,
                                   kaldi::BaseFloat sample_rate)
{
    if (fbank_computer == nullptr)
    {
        KALDI_ERR << "fbank_computer is nullptr";
        return;
    }

    try
    {
        fbank_computer->ComputeFeatures(signals, sample_rate, 1.0, output);
        if (subtract_mean)
        {
            kaldi::Vector<kaldi::BaseFloat> mean(output->NumCols());
            mean.AddRowSumMat(1.0, *output);
            mean.Scale(1.0 / output->NumRows());
            for (kaldi::int32 i = 0; i < output->NumRows(); i++)
                output->Row(i).AddVec(-1.0, mean);
        }
    }
    catch (...)
    {
        KALDI_WARN << "Failed to compute features for signals";
    }
}
