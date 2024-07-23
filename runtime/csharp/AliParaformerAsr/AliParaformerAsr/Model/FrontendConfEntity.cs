// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace AliParaformerAsr.Model
{
    public class FrontendConfEntity
    {
        private int _fs = 16000;
        private string _window = "hamming";
        private int _n_mels = 80;
        private int _frame_length = 25;
        private int _frame_shift = 10;
        private float _dither = 1.0F;
        private int _lfr_m = 7;
        private int _lfr_n = 6;
        private bool _snip_edges = false;

        public int fs { get => _fs; set => _fs = value; }
        public string window { get => _window; set => _window = value; }
        public int n_mels { get => _n_mels; set => _n_mels = value; }
        public int frame_length { get => _frame_length; set => _frame_length = value; }
        public int frame_shift { get => _frame_shift; set => _frame_shift = value; }
        public float dither { get => _dither; set => _dither = value; }
        public int lfr_m { get => _lfr_m; set => _lfr_m = value; }
        public int lfr_n { get => _lfr_n; set => _lfr_n = value; }
        public bool snip_edges { get => _snip_edges; set => _snip_edges = value; }
    }
}
