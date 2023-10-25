using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using AliFsmnVadSharp.Struct;

namespace AliFsmnVadSharp.DLL
{
    public static class KaldiNativeFbank
    {
        private const string dllName = @"kaldi-native-fbank-dll";

        [DllImport(dllName, EntryPoint = "GetFbankOptions", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr GetFbankOptions(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10.0f, float frame_length = 25.0f, float energy_floor = 0.0f, bool debug_mel = false, string window_type = "hamming");

        [DllImport(dllName, EntryPoint = "GetOnlineFbank", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern KnfOnlineFbank GetOnlineFbank(IntPtr opts);

        [DllImport(dllName, EntryPoint = "AcceptWaveform", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void AcceptWaveform(KnfOnlineFbank knfOnlineFbank, float sample_rate, float[] samples, int samples_size);

        [DllImport(dllName, EntryPoint = "InputFinished", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void InputFinished(KnfOnlineFbank knfOnlineFbank);

        [DllImport(dllName, EntryPoint = "GetNumFramesReady", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GetNumFramesReady(KnfOnlineFbank knfOnlineFbank);

        [DllImport(dllName, EntryPoint = "AcceptWaveformxxx", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern FbankDatas AcceptWaveformxxx(KnfOnlineFbank knfOnlineFbank, float sample_rate, float[] samples, int samples_size);

        [DllImport(dllName, EntryPoint = "GetFbank", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void GetFbank(KnfOnlineFbank knfOnlineFbank,int frame, ref FbankData pData);

        [DllImport(dllName, EntryPoint = "GetFbanks", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void GetFbanks(KnfOnlineFbank knfOnlineFbank, int framesNum, ref FbankDatas fbankDatas);
        
    }
}
