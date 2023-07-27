// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVad.Model
{
    internal class VadInputEntity
    {
        private float[]? _speech;
        private int _speechLength;
        private List<float[]> _inCaches = new List<float[]>();
        private float[]? _waveform;
        private E2EVadModel _vad_scorer;

        public float[]? Speech { get => _speech; set => _speech = value; }
        public int SpeechLength { get => _speechLength; set => _speechLength = value; }
        public List<float[]> InCaches { get => _inCaches; set => _inCaches = value; }
        public float[] Waveform { get => _waveform; set => _waveform = value; }
        internal E2EVadModel VadScorer { get => _vad_scorer; set => _vad_scorer = value; }
    }
}
