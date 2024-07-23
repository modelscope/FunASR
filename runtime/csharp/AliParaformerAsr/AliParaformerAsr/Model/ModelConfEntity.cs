// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace AliParaformerAsr.Model
{
    public class ModelConfEntity
    {
        private float _ctc_weight = 0.0F;
        private float _lsm_weight = 0.1F;
        private bool _length_normalized_loss = true;
        private float _predictor_weight = 1.0F;
        private int _predictor_bias = 1;
        private float _sampling_ratio = 0.75F;
        private int _sos = 1;
        private int _eos = 2;
        private int _ignore_id = -1;

        public float ctc_weight { get => _ctc_weight; set => _ctc_weight = value; }
        public float lsm_weight { get => _lsm_weight; set => _lsm_weight = value; }
        public bool length_normalized_loss { get => _length_normalized_loss; set => _length_normalized_loss = value; }
        public float predictor_weight { get => _predictor_weight; set => _predictor_weight = value; }
        public int predictor_bias { get => _predictor_bias; set => _predictor_bias = value; }
        public float sampling_ratio { get => _sampling_ratio; set => _sampling_ratio = value; }
        public int sos { get => _sos; set => _sos = value; }
        public int eos { get => _eos; set => _eos = value; }
        public int ignore_id { get => _ignore_id; set => _ignore_id = value; }
    }
}
