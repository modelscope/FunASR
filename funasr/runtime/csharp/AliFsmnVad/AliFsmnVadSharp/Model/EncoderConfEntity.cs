using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVadSharp.Model
{
    public class EncoderConfEntity
    {
        private int _input_dim=400;
        private int _input_affineDim = 140;
        private int _fsmn_layers = 4;
        private int _linear_dim = 250;
        private int _proj_dim = 128;
        private int _lorder = 20;
        private int _rorder = 0;
        private int _lstride = 1;
        private int _rstride = 0;
        private int _output_dffine_dim = 140;
        private int _output_dim = 248;

        public int input_dim { get => _input_dim; set => _input_dim = value; }
        public int input_affine_dim { get => _input_affineDim; set => _input_affineDim = value; }
        public int fsmn_layers { get => _fsmn_layers; set => _fsmn_layers = value; }
        public int linear_dim { get => _linear_dim; set => _linear_dim = value; }
        public int proj_dim { get => _proj_dim; set => _proj_dim = value; }
        public int lorder { get => _lorder; set => _lorder = value; }
        public int rorder { get => _rorder; set => _rorder = value; }
        public int lstride { get => _lstride; set => _lstride = value; }
        public int rstride { get => _rstride; set => _rstride = value; }
        public int output_affine_dim { get => _output_dffine_dim; set => _output_dffine_dim = value; }
        public int output_dim { get => _output_dim; set => _output_dim = value; }
    }
}
