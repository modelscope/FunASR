using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVadSharp.Model
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
