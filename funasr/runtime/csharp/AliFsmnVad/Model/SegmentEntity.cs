// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVad.Model
{
    public class SegmentEntity
    {
        private List<int[]> _segment=new List<int[]>();
        private List<float[]> _waveform=new List<float[]>();

        public List<int[]> Segment { get => _segment; set => _segment = value; }
        public List<float[]> Waveform { get => _waveform; set => _waveform = value; }
    }
}
