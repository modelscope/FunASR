using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVadSharp.Model
{
    public class FrontendConfEntity
    {
        private int _fs = 16000;
        private string _window = "hamming";
        private int _n_mels = 80;
        private int _frame_length = 25;
        private int _frame_shift = 10;
        private float _dither = 0.0F;
        private int _lfr_m = 5;
        private int _lfr_n = 1;

        public int fs { get => _fs; set => _fs = value; }
        public string window { get => _window; set => _window = value; }
        public int n_mels { get => _n_mels; set => _n_mels = value; }
        public int frame_length { get => _frame_length; set => _frame_length = value; }
        public int frame_shift { get => _frame_shift; set => _frame_shift = value; }
        public float dither { get => _dither; set => _dither = value; }
        public int lfr_m { get => _lfr_m; set => _lfr_m = value; }
        public int lfr_n { get => _lfr_n; set => _lfr_n = value; }
    }
}
