using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliCTTransformerPunc.Model
{
    /// <summary>
    /// PuncInputEntity
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class PuncInputEntity
    {
        private int[] _miniSentenceId;
        private int _textLengths;

        public int[] MiniSentenceId { get => _miniSentenceId; set => _miniSentenceId = value; }
        public int TextLengths { get => _textLengths; set => _textLengths = value; }
    }
}
