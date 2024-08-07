// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using AliParaformerAsr.Model;
using Microsoft.ML.OnnxRuntime;

namespace AliParaformerAsr
{
    internal interface IOfflineProj
    {
        InferenceSession ModelSession 
        {
            get;
            set;
        }
        int Blank_id
        {
            get;
            set;
        }
        int Sos_eos_id
        {
            get;
            set;
        }
        int Unk_id
        {
            get;
            set;
        }
        int SampleRate
        {
            get;
            set;
        }
        int FeatureDim
        {
            get;
            set;
        }
        internal ModelOutputEntity ModelProj(List<OfflineInputEntity> modelInputs);
        internal void Dispose();
    }
}
