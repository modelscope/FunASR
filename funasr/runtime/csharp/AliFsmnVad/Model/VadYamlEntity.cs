// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVad.Model
{
    internal class VadYamlEntity
    {
        private int _input_size;
        private string _frontend = "wav_frontend";
        private FrontendConfEntity _frontend_conf=new FrontendConfEntity();
        private string _model = "e2evad";
        private string _encoder = "fsmn";
        private EncoderConfEntity _encoder_conf=new EncoderConfEntity();
        private VadPostConfEntity _vad_post_conf=new VadPostConfEntity();

        public int input_size { get => _input_size; set => _input_size = value; }
        public string frontend { get => _frontend; set => _frontend = value; }
        public string model { get => _model; set => _model = value; }
        public string encoder { get => _encoder; set => _encoder = value; }
        public FrontendConfEntity frontend_conf { get => _frontend_conf; set => _frontend_conf = value; }
        public EncoderConfEntity encoder_conf { get => _encoder_conf; set => _encoder_conf = value; }
        public VadPostConfEntity vad_post_conf { get => _vad_post_conf; set => _vad_post_conf = value; }
    }
}
