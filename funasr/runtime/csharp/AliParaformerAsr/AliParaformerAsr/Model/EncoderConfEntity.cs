// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliParaformerAsr.Model
{
    public class EncoderConfEntity
    {
        private int _output_size = 512;
        private int _attention_heads = 4;
        private int _linear_units = 2048;
        private int _num_blocks = 50;
        private float _dropout_rate = 0.1F;
        private float _positional_dropout_rate = 0.1F;
        private float _attention_dropout_rate= 0.1F;
        private string _input_layer = "pe";
        private string _pos_enc_class = "SinusoidalPositionEncoder";
        private bool _normalize_before = true;
        private int _kernel_size = 11;
        private int _sanm_shfit = 0;
        private string _selfattention_layer_type = "sanm";

        public int output_size { get => _output_size; set => _output_size = value; }
        public int attention_heads { get => _attention_heads; set => _attention_heads = value; }
        public int linear_units { get => _linear_units; set => _linear_units = value; }
        public int num_blocks { get => _num_blocks; set => _num_blocks = value; }
        public float dropout_rate { get => _dropout_rate; set => _dropout_rate = value; }
        public float positional_dropout_rate { get => _positional_dropout_rate; set => _positional_dropout_rate = value; }
        public float attention_dropout_rate { get => _attention_dropout_rate; set => _attention_dropout_rate = value; }
        public string input_layer { get => _input_layer; set => _input_layer = value; }
        public string pos_enc_class { get => _pos_enc_class; set => _pos_enc_class = value; }
        public bool normalize_before { get => _normalize_before; set => _normalize_before = value; }
        public int kernel_size { get => _kernel_size; set => _kernel_size = value; }
        public int sanm_shfit { get => _sanm_shfit; set => _sanm_shfit = value; }
        public string selfattention_layer_type { get => _selfattention_layer_type; set => _selfattention_layer_type = value; }
    }
}
