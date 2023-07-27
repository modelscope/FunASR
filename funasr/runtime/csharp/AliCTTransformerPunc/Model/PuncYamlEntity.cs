using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliCTTransformerPunc.Model
{
    /// <summary>
    /// PuncYamlEntity
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class PuncYamlEntity
    {
        private int _init = int.MinValue;
        private ModelConfEntity _model_conf=new ModelConfEntity();
        private bool _use_preprocessor = true;
        private string _token_type = "word";
        private string _bpemodel = String.Empty;
        private string _non_linguistic_symbols = String.Empty;
        private string _cleaner = String.Empty;
        private string _g2p = String.Empty;
        private string _punctuation = "target_delay";
        private PunctuationConfEntity _punctuation_conf=new PunctuationConfEntity();
        private int _gpu_id = 0;
        private string[] _punc_list = new string[] { "<unk>", "_", "','", "。", "'?'", "、" };
        private bool _distributed = true;
        private string _version = "0.1.7";

        public int init { get => _init; set => _init = value; }
        public bool use_preprocessor { get => _use_preprocessor; set => _use_preprocessor = value; }
        public string token_type { get => _token_type; set => _token_type = value; }
        public string bpemodel { get => _bpemodel; set => _bpemodel = value; }
        public string non_linguistic_symbols { get => _non_linguistic_symbols; set => _non_linguistic_symbols = value; }
        public string cleaner { get => _cleaner; set => _cleaner = value; }
        public string g2p { get => _g2p; set => _g2p = value; }
        public string punctuation { get => _punctuation; set => _punctuation = value; }
        public int gpu_id { get => _gpu_id; set => _gpu_id = value; }
        public string[] punc_list { get => _punc_list; set => _punc_list = value; }
        public bool distributed { get => _distributed; set => _distributed = value; }
        public string version { get => _version; set => _version = value; }
        public ModelConfEntity model_conf { get => _model_conf; set => _model_conf = value; }
        public PunctuationConfEntity punctuation_conf { get => _punctuation_conf; set => _punctuation_conf = value; }
    }
}
