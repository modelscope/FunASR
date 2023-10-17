// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliParaformerAsr.Model
{
    public class PredictorConfEntity
    {
        private int _idim = 512;
        private float _threshold = 1.0F;
        private int _l_order = 1;
        private int _r_order = 1;
        private float _tail_threshold = 0.45F;

        public int idim { get => _idim; set => _idim = value; }
        public float threshold { get => _threshold; set => _threshold = value; }
        public int l_order { get => _l_order; set => _l_order = value; }
        public int r_order { get => _r_order; set => _r_order = value; }
        public float tail_threshold { get => _tail_threshold; set => _tail_threshold = value; }
    }
}
