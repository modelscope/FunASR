using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliCTTransformerPunc.Model
{
    /// <summary>
    /// PuncOutputEntity
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class PuncOutputEntity
    {
        private float[]? logits;
        private List<int[]>? _punctuations = new List<int[]>() { new int[4] };

        public float[]? Logits { get => logits; set => logits = value; }
        public List<int[]>? Punctuations { get => _punctuations; set => _punctuations = value; }
    }
}
