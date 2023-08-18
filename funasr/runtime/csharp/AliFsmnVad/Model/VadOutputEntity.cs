// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVad.Model
{
    internal class VadOutputEntity
    {
        private float[,,]? _scores;
        private List<float[]> _outCaches=new List<float[]>();
        private float[]? _waveform;

        public float[,,]? Scores { get => _scores; set => _scores = value; }
        public List<float[]> OutCaches { get => _outCaches; set => _outCaches = value; }
        public float[] Waveform { get => _waveform; set => _waveform = value; }
    }
}
