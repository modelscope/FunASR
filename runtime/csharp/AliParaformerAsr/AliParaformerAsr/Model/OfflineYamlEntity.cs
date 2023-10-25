// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliParaformerAsr.Model
{
    internal class OfflineYamlEntity
    {
        private int _input_size;
        private string _frontend = "wav_frontend";
        private FrontendConfEntity _frontend_conf = new FrontendConfEntity();
        private string _model = "paraformer";
        private ModelConfEntity _model_conf = new ModelConfEntity();
        private string _preencoder = string.Empty;
        private PostEncoderConfEntity _preencoder_conf = new PostEncoderConfEntity();
        private string _encoder = "sanm";
        private EncoderConfEntity _encoder_conf = new EncoderConfEntity();
        private string _postencoder = string.Empty;
        private PostEncoderConfEntity _postencoder_conf = new PostEncoderConfEntity();
        private string _decoder = "paraformer_decoder_sanm";
        private DecoderConfEntity _decoder_conf = new DecoderConfEntity();
        private string _predictor = "cif_predictor_v2";
        private PredictorConfEntity _predictor_conf = new PredictorConfEntity();
        private string _version = string.Empty;


        public int input_size { get => _input_size; set => _input_size = value; }
        public string frontend { get => _frontend; set => _frontend = value; }
        public FrontendConfEntity frontend_conf { get => _frontend_conf; set => _frontend_conf = value; }
        public string model { get => _model; set => _model = value; }
        public ModelConfEntity model_conf { get => _model_conf; set => _model_conf = value; }
        public string preencoder { get => _preencoder; set => _preencoder = value; }
        public PostEncoderConfEntity preencoder_conf { get => _preencoder_conf; set => _preencoder_conf = value; }
        public string encoder { get => _encoder; set => _encoder = value; }
        public EncoderConfEntity encoder_conf { get => _encoder_conf; set => _encoder_conf = value; }
        public string postencoder { get => _postencoder; set => _postencoder = value; }
        public PostEncoderConfEntity postencoder_conf { get => _postencoder_conf; set => _postencoder_conf = value; }
        public string decoder { get => _decoder; set => _decoder = value; }
        public DecoderConfEntity decoder_conf { get => _decoder_conf; set => _decoder_conf = value; }
        public string predictor { get => _predictor; set => _predictor = value; }
        public string version { get => _version; set => _version = value; }
        public PredictorConfEntity predictor_conf { get => _predictor_conf; set => _predictor_conf = value; }
    }
}
