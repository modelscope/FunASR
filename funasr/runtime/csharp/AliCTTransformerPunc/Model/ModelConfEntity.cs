using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliCTTransformerPunc.Model
{
    /// <summary>
    /// ModelConfEntity
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class ModelConfEntity
    {
        private int _ignore_id = 0;

        public int ignore_id { get => _ignore_id; set => _ignore_id = value; }
    }
}
