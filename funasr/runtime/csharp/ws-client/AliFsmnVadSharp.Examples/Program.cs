using AliFsmnVadSharp;
using AliFsmnVadSharp.Model;
using Microsoft.Extensions.Logging;
using NAudio.Wave;
using NAudio.Wave.Compression;
using NAudio.Wave.SampleProviders;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Xml.Linq;
using YamlDotNet.Core.Tokens;
using WebSocketSpace;
using NAudio.CoreAudioApi;
using System.Runtime.InteropServices;
using System.Linq;

namespace FunASRClient
{
    /// <summary>
    /// /主程序入口
    /// </summary>
    public class Program
    {
        private static void Main()
        {
            FunASR_Stream_Client m_funasrclient = new FunASR_Stream_Client();
            m_funasrclient.FunASR_Main();
        }
    }
    /// <summary>
    /// /主线程入口，初始化后读取数据
    /// </summary>
    public class FunASR_Stream_Client
    {
        /// <summary>
        /// FunASR客户端软件运行状态
        /// </summary>
        public static bool FunASRClientIsRunning = true;
        private static byte[] cur_audio_buff = null;
        private static object lockObj = new object();
        private static WaveCollect m_wavecollect = new WaveCollect();
        private static CWebSocketClient m_websocketclient = new CWebSocketClient();
        public static readonly ConcurrentQueue<byte[]> ActiveAudioSet = new ConcurrentQueue<byte[]>();
        public static readonly ConcurrentQueue<string> AudioFileQueue = new ConcurrentQueue<string>();
        [STAThread]
        public async void FunASR_Main()
        {
            //麦克风状态监测
            string errorStatus = string.Empty;
            if (GetCurrentMicVolume() == -2)
                errorStatus = "注意：麦克风被设置为静音！";
            else if (GetCurrentMicVolume() == -1)
                errorStatus = "注意：麦克风未连接！";
            else if (GetCurrentMicVolume() == 0)
                errorStatus = "注意：麦克风声音设置为0！";

            //开启实时VAD监测线程
            Thread VADThread = new Thread(FsmnVadTheAudio);
            VADThread.Start();
            //初始化通信连接
            string commstatus = ClientConnTest();
            if (commstatus != "通信连接成功")
                errorStatus = commstatus;

            //程序初始监测异常--报错、退出
            if (errorStatus != string.Empty)
            {
                //报错方式待加
                Environment.Exit(0);
            }

            //启动客户端向服务端发送音频数据线程
            Thread SendAudioThread = new Thread(SendAudioToSeverAsync);
            SendAudioThread.Start();

            //启动音频文件转录线程
            Thread AudioFileThread = new Thread(SendAudioFileToSeverAsync);
            AudioFileThread.Start();

            AudioFileQueue.Enqueue(AppDomain.CurrentDomain.BaseDirectory + @"speech_fsmn_vad_zh-cn-16k-common-pytorch\example\0.wav");
            AudioFileQueue.Enqueue(AppDomain.CurrentDomain.BaseDirectory + @"speech_fsmn_vad_zh-cn-16k-common-pytorch\example\1.wav");
            AudioFileQueue.Enqueue(AppDomain.CurrentDomain.BaseDirectory + @"speech_fsmn_vad_zh-cn-16k-common-pytorch\example\2.wav");
            AudioFileQueue.Enqueue(AppDomain.CurrentDomain.BaseDirectory + @"speech_fsmn_vad_zh-cn-16k-common-pytorch\example\3.wav");
            AudioFileQueue.Enqueue(AppDomain.CurrentDomain.BaseDirectory + @"speech_fsmn_vad_zh-cn-16k-common-pytorch\example\4.wav");
            AudioFileQueue.Enqueue(AppDomain.CurrentDomain.BaseDirectory + @"speech_fsmn_vad_zh-cn-16k-common-pytorch\example\5.wav");
            //开启声音录制
            m_wavecollect.StartRec();
            try
            {
                while (FunASRClientIsRunning)
                {
                    if (!WaveCollect.voicebuff.IsEmpty)
                    {
                        byte[] buff;
                        int buffcnt = WaveCollect.voicebuff.Count;
                        WaveCollect.voicebuff.TryDequeue(out buff);

                        //if (data != null)
                        //    Console.WriteLine(BitConverter.ToString(data));
                        if (buff == null)
                            continue;

                        lock (lockObj)
                        {
                            if (cur_audio_buff == null)
                            {
                                cur_audio_buff = buff;
                            }
                            else
                            {
                                byte[] mArray = new byte[cur_audio_buff.Length + buff.Length];
                                Array.Copy(cur_audio_buff, 0, mArray, 0, cur_audio_buff.Length);
                                Buffer.BlockCopy(buff, 0, mArray, cur_audio_buff.Length, buff.Length);
                                cur_audio_buff = mArray;
                            }
                        }
                    }
                    else
                    {
                        Thread.Sleep(1);
                    }
                }
            }
            finally//程序退出
            {
                m_wavecollect.StopRec();
            }
        }
        private static string ClientConnTest()
        {
            //WebSocket连接状态监测
            Task<string> websocketstatus = m_websocketclient.ClientConnTest();
            if (websocketstatus != null && websocketstatus.Result.IndexOf("成功") == -1)
                return websocketstatus.Result;
            return "通信连接成功";
        }
        void FsmnVadTheAudio()
        {
            //读取模型文件地址
            string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
            string modelFilePath = applicationBase + @"speech_fsmn_vad_zh-cn-16k-common-pytorch\model.onnx";
            string configFilePath = applicationBase + @"speech_fsmn_vad_zh-cn-16k-common-pytorch\vad.yaml";
            string mvnFilePath = applicationBase + @"speech_fsmn_vad_zh-cn-16k-common-pytorch\vad.mvn";
            //初始化基本参数
            int batchSize = 2;
            int cur_audio_cnt = 0;
            int record_mic_count = 0;
            TimeSpan start_time0 = new TimeSpan(DateTime.Now.Ticks);
            AliFsmnVad aliFsmnVad = new AliFsmnVad(modelFilePath, configFilePath, mvnFilePath, batchSize);
            TimeSpan end_time0 = new TimeSpan(DateTime.Now.Ticks);
            //写入模型初始化的时间
            double elapsed_milliseconds0 = end_time0.TotalMilliseconds - start_time0.TotalMilliseconds;
            Console.WriteLine("load model and init config elapsed_milliseconds:{0}", elapsed_milliseconds0.ToString());
            Dictionary<int, int> maps = new Dictionary<int, int>();  //记录最新缓存
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks); //记录开始时间
            while (FunASRClientIsRunning)
            {
                lock (lockObj)//优化线程数据
                {
                    if (cur_audio_buff != null && cur_audio_buff.Length > 12800 && cur_audio_cnt != cur_audio_buff.Length)
                    {
                        cur_audio_cnt = cur_audio_buff.Length;
                        var pcmProvider = new BufferedWaveProvider(new WaveFormat(WaveCollect.wave_buffer_collectfrequency
                               , WaveCollect.wave_buffer_collectbits, WaveCollect.wave_buffer_collectchannels));
                        int secaudio = (WaveCollect.wave_buffer_collectfrequency * WaveCollect.wave_buffer_collectchannels
                                    * WaveCollect.wave_buffer_collectbits / 8);
                        pcmProvider.BufferDuration = TimeSpan.FromSeconds(cur_audio_buff.Length / secaudio + 1);
                        byte[] resultArray = new byte[cur_audio_buff.Length];
                        Array.Copy(cur_audio_buff, 0, resultArray, 0, cur_audio_buff.Length);
                        pcmProvider.AddSamples(resultArray, 0, resultArray.Length);
                        ISampleProvider sampleprovider = pcmProvider.ToSampleProvider();

                        float[] sample = new float[resultArray.Length];
                        sampleprovider.Read(sample, 0, resultArray.Length);
                        pcmProvider.ClearBuffer();
                        List<float[]> samples = new List<float[]>();
                        samples.Add(sample);

                        Dictionary<int, int> map = new Dictionary<int, int>();
                        SegmentEntity[] segments_duration = aliFsmnVad.GetSegments(samples);
                        samples.Reverse();
                        //SegmentEntity[] segments_duration = aliFsmnVad.GetSegmentsByStep(samples);
                        TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                        //Console.WriteLine("vad infer result:");
                        foreach (SegmentEntity segment in segments_duration)
                        {
                            if (segment != null)
                            {
                                Console.Write("[");
                                foreach (var x in segment.Segment)
                                {
                                    for (int pos = 0; pos < x.Length / 2; pos = pos + 2)
                                    {
                                        map.Add(x[pos], x[pos + 1]);
                                    }
                                    Console.Write("[" + string.Join(",", x.ToArray()) + "]");
                                }
                                Console.Write("]\r\n");
                            }
                        }
                        long max_audio_pos = 0;
                        foreach (var item in map)
                        {
                            if (map.Count == 1 && item.Key == 0 && item.Value == 10)
                            {
                                continue;
                            }
                            else//截取其中片段-(按段取出)
                            {
                                //判断当前语音是否已经截断
                                if (maps.ContainsKey(item.Key))
                                {
                                    int value = maps[item.Key];
                                    if (value == item.Value)
                                    {
                                        //知识科普
                                        //1秒钟总数据量 = 音频采样率 * 采样通道 * 位深度 / 8;
                                        //每帧音频数据大小 = 1秒钟总数据量 / 1秒钟采样次数 = 1秒钟总数据量 / (1秒钟 / 采用频率)
                                        long partbegin = item.Key * (WaveCollect.wave_buffer_collectfrequency * WaveCollect.wave_buffer_collectchannels
                                            * WaveCollect.wave_buffer_collectbits / 8) / 1000;
                                        long partend = item.Value * (WaveCollect.wave_buffer_collectfrequency * WaveCollect.wave_buffer_collectchannels
                                            * WaveCollect.wave_buffer_collectbits / 8) / 1000;
                                        if (partend > max_audio_pos)
                                            max_audio_pos = partend;
                                        //读取有效音频缓存
                                        if (cur_audio_buff.Length < partend)
                                            partend = cur_audio_buff.Length;

                                        byte[] ActiveAudio = new byte[partend - partbegin];
                                        Array.Copy(cur_audio_buff, partbegin, ActiveAudio, 0, partend - partbegin);
                                        ActiveAudioSet.Enqueue(ActiveAudio);

                                        maps.Remove(item.Key);
                                        record_mic_count++;
                                        //保留VAD语音片段到文件夹
                                        //string savewavepath = string.Format(applicationBase + "record\\{0}.wav", record_mic_count);
                                        //SaveAsWav(ActiveAudio, savewavepath, WaveCollect.wave_buffer_collectfrequency,
                                        //    WaveCollect.wave_buffer_collectbits, WaveCollect.wave_buffer_collectchannels);
                                    }
                                    else
                                    {
                                        maps[item.Key] = item.Value;
                                    }
                                }
                                else
                                {
                                    maps.Add(item.Key, item.Value);
                                }
                            }
                        }
                        //音频重设后，重置maps
                        foreach (var item in maps)
                        {
                            if (!map.ContainsKey(item.Key))
                                maps.Remove((int)item.Key);
                        }
                        //重设音频存储区数据
                        if (max_audio_pos > 0)
                        {
                            if (max_audio_pos > cur_audio_buff.Length)
                                max_audio_pos = cur_audio_buff.Length;
                            byte[] mArray = new byte[cur_audio_buff.Length - max_audio_pos];
                            Array.Copy(cur_audio_buff, max_audio_pos, mArray, 0, cur_audio_buff.Length - max_audio_pos);
                            cur_audio_buff = mArray;
                        }
                        else if (cur_audio_buff.Length > (WaveCollect.wave_buffer_collectfrequency * WaveCollect.wave_buffer_collectchannels
                                    * WaveCollect.wave_buffer_collectbits / 8) * 60)
                        {
                            int m_60Secaudio = (WaveCollect.wave_buffer_collectfrequency * WaveCollect.wave_buffer_collectchannels
                                    * WaveCollect.wave_buffer_collectbits / 8) * 60;
                            //防止空音频过多导致内存爆炸
                            if ((maps.Count == 0) || (maps.Count == 1 && maps.ContainsKey(0) && maps[0] == 10))
                            {
                                int m_40Secaudio = m_60Secaudio / 6 * 4;//60S空音频删除前40S
                                byte[] mArray = new byte[cur_audio_buff.Length - m_40Secaudio];
                                Array.Copy(cur_audio_buff, m_40Secaudio, mArray, 0, cur_audio_buff.Length - m_40Secaudio);
                                cur_audio_buff = mArray;
                            }
                        }
                        GC.Collect();
                        //double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                        //Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
                        //Console.WriteLine("WaveCollect_voicebuff_count:{0}", WaveCollect.voicebuff.Count);
                        //Console.WriteLine("------------------------");
                    }
                    else
                    {
                        Thread.Sleep(10);
                    }
                }
            }
        }
        async void SendAudioFileToSeverAsync()
        {
            while (FunASRClientIsRunning)
            {
                Thread.Sleep(1000);
                if (AudioFileQueue.Count > 0)
                {
                    string filepath = string.Empty;
                    AudioFileQueue.TryDequeue(out filepath);
                    if (filepath != string.Empty && filepath != null)
                    {
                        await m_websocketclient.ClientSendFileFunc(filepath);
                        GC.Collect();
                    }
                }
                else
                {
                    Thread.Sleep(100);
                }
            }
        }
        async void SendAudioToSeverAsync()
        {
            while (FunASRClientIsRunning)
            {
                if (ActiveAudioSet.Count > 0)
                {
                    byte[] audio;
                    ActiveAudioSet.TryDequeue(out audio);
                    if (audio == null)
                        continue;

                    byte[] mArray = new byte[audio.Length];
                    Array.Copy(audio, 0, mArray, 0, audio.Length);
                    if (mArray != null)
                        await m_websocketclient.ClientSendAudioFunc(mArray);
                    GC.Collect();
                }
                else
                {
                    Thread.Sleep(10);
                }
            }
        }

        private void SaveAsWav(byte[] pcmData, string fileName, int sampleRate, int bitsPerSample, int channels)
        {
            using (var writer = new WaveFileWriter(fileName, new WaveFormat(sampleRate, bitsPerSample, channels)))
            {
                writer.Write(pcmData, 0, pcmData.Length);
            }
        }

        private int GetCurrentMicVolume()                       //获取麦克风设置
        {
            int volume = -1;
            var enumerator = new MMDeviceEnumerator();

            //获取音频输入设备
            IEnumerable<MMDevice> captureDevices = enumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active).ToArray();
            if (captureDevices.Count() > 0)
            {
                MMDevice mMDevice = captureDevices.ToList()[0];
                if (mMDevice.AudioEndpointVolume.Mute)
                    return -2;
                volume = (int)(mMDevice.AudioEndpointVolume.MasterVolumeLevelScalar * 100);

            }
            return volume;
        }
    }
}