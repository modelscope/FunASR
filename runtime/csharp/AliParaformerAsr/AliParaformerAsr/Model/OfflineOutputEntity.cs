// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace AliParaformerAsr.Model
{
    public class OfflineOutputEntity
    {

        private float[]? logits;
        private long[]? _token_num;
        private List<int[]>? _token_nums=new List<int[]>() { new int[4]};
        private int[] _token_nums_length;

        public float[]? Logits { get => logits; set => logits = value; }
        public long[]? Token_num { get => _token_num; set => _token_num = value; }
        public List<int[]>? Token_nums { get => _token_nums; set => _token_nums = value; }
        public int[] Token_nums_length { get => _token_nums_length; set => _token_nums_length = value; }
    }
}
