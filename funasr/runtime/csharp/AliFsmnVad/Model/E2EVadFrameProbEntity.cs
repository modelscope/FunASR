// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVad.Model
{
    internal class E2EVadFrameProbEntity
    {
        private double _noise_prob = 0.0F;
        private double _speech_prob = 0.0F;
        private double _score = 0.0F;
        private int _frame_id = 0;
        private int _frm_state = 0;

        public double noise_prob { get => _noise_prob; set => _noise_prob = value; }
        public double speech_prob { get => _speech_prob; set => _speech_prob = value; }
        public double score { get => _score; set => _score = value; }
        public int frame_id { get => _frame_id; set => _frame_id = value; }
        public int frm_state { get => _frm_state; set => _frm_state = value; }
    }
}
