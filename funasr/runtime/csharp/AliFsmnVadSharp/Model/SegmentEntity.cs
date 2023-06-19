using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVadSharp.Model
{
    public class SegmentEntity
    {
        private List<int[]> _segment=new List<int[]>();
        private List<float[]> _waveform=new List<float[]>();

        public List<int[]> Segment { get => _segment; set => _segment = value; }
        public List<float[]> Waveform { get => _waveform; set => _waveform = value; }
        //public SegmentEntity()
        //{
        //    int[] t=new int[0];
        //    _segment.Add(t);
        //}
    }
}
