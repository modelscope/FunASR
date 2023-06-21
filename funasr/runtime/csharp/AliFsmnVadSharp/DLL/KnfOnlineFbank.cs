using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVadSharp.DLL
{
    internal struct FbankData
    {
        public IntPtr data;
        public int data_length;
    };

    internal struct FbankDatas
    {
        public IntPtr data;
        public int data_length;
    };

    internal struct KnfOnlineFbank
    {
        public IntPtr impl;
    };
}
