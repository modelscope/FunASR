// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliParaformerAsr.Model
{
    public class DecoderConfEntity
    {
        private int _attention_heads = 4;
        private int _linear_units = 2048;
        private int _num_blocks = 16;
        private float _dropout_rate = 0.1F;
        private float _positional_dropout_rate = 0.1F;
        private float _self_attention_dropout_rate= 0.1F;
        private float _src_attention_dropout_rate = 0.1F;
        private int _att_layer_num = 16;
        private int _kernel_size = 11;
        private int _sanm_shfit = 0;

        public int attention_heads { get => _attention_heads; set => _attention_heads = value; }
        public int linear_units { get => _linear_units; set => _linear_units = value; }
        public int num_blocks { get => _num_blocks; set => _num_blocks = value; }
        public float dropout_rate { get => _dropout_rate; set => _dropout_rate = value; }
        public float positional_dropout_rate { get => _positional_dropout_rate; set => _positional_dropout_rate = value; }
        public float self_attention_dropout_rate { get => _self_attention_dropout_rate; set => _self_attention_dropout_rate = value; }
        public float src_attention_dropout_rate { get => _src_attention_dropout_rate; set => _src_attention_dropout_rate = value; }
        public int att_layer_num { get => _att_layer_num; set => _att_layer_num = value; }
        public int kernel_size { get => _kernel_size; set => _kernel_size = value; }
        public int sanm_shfit { get => _sanm_shfit; set => _sanm_shfit = value; }
        
    }
}
