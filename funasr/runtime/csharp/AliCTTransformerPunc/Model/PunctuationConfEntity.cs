using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliCTTransformerPunc.Model
{
    /// <summary>
    /// PunctuationConfEntity
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class PunctuationConfEntity
    {
        private string _pos_enc = "sinusoidal";
        private int _embed_unit = 256;
        private int _att_unit = 256;
        private int _head = 8;
        private int _unit = 1024;
        private int _layer = 4;
        private float _dropout_rate = 0.1F;

        public string pos_enc { get => _pos_enc; set => _pos_enc = value; }
        public int embed_unit { get => _embed_unit; set => _embed_unit = value; }
        public int att_unit { get => _att_unit; set => _att_unit = value; }
        public int head { get => _head; set => _head = value; }
        public int unit { get => _unit; set => _unit = value; }
        public int layer { get => _layer; set => _layer = value; }
        public float dropout_rate { get => _dropout_rate; set => _dropout_rate = value; }
    }
}
