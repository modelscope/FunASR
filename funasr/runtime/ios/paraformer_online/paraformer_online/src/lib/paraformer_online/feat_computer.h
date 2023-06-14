/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

#ifndef FEATURES_H_
#define FEATURES_H_
#include "base/kaldi-common.h"
#include "feat/feature-fbank.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"

class FeatComputer
{
public:
    FeatComputer();
    void compute_signals(kaldi::SubVector<kaldi::BaseFloat> signals,
                                       kaldi::Matrix<kaldi::BaseFloat> *Output,
                                       kaldi::BaseFloat sample_rate);
    ~FeatComputer()
    {
        if (fbank_computer)
        {
            delete fbank_computer;
            fbank_computer = nullptr;
        }
    }

private:
    kaldi::FbankOptions fbank_opts;
    kaldi::Fbank *fbank_computer;
    bool subtract_mean = false;
};

#endif
