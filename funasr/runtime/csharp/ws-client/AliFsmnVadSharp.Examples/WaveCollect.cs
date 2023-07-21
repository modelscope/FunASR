using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NAudio.CoreAudioApi;

namespace AliFsmnVadSharp
{
    class WaveCollect
    {
        private string fileName = string.Empty;
        private WaveInEvent? waveSource = null;
        private WaveFileWriter? waveFile = null;
        public static int wave_buffer_milliseconds = 500;
        public static int wave_buffer_collectbits = 16;
        public static int wave_buffer_collectchannels = 1;
        public static int wave_buffer_collectfrequency = 16000;
        public static readonly ConcurrentQueue<byte[]> voicebuff = new ConcurrentQueue<byte[]>();

        public void StartRec()
        {
            // 获取麦克风设备
            var captureDevices = new MMDeviceEnumerator().EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active);
            foreach (var device in captureDevices)
            {
                Console.WriteLine("Device Name: " + device.FriendlyName);
                using (var capture = new WasapiLoopbackCapture(device))
                {
                    // 获取支持的采样率列表
                    Console.WriteLine("Device Channels:" + capture.WaveFormat.Channels);
                    Console.WriteLine("Device SampleRate:" + capture.WaveFormat.SampleRate);
                    Console.WriteLine("Device BitsPerSample:" + capture.WaveFormat.BitsPerSample);
                }
            }
            //清空缓存数据
            int buffnum = voicebuff.Count;
            for (int i = 0; i < buffnum; i++)
                voicebuff.TryDequeue(out byte[] buff);

            waveSource = new WaveInEvent();
            waveSource.BufferMilliseconds = wave_buffer_milliseconds;
            waveSource.WaveFormat = new WaveFormat(wave_buffer_collectfrequency, wave_buffer_collectbits, wave_buffer_collectchannels); // 16bit,16KHz,Mono的录音格式
            waveSource.DataAvailable += new EventHandler<WaveInEventArgs>(waveSource_DataAvailable);
            SetFileName(AppDomain.CurrentDomain.BaseDirectory + "wav\\tmp.wav");
            waveFile = new WaveFileWriter(fileName, waveSource.WaveFormat);
            waveSource.StartRecording();
        }

        public void StopRec()
        {
            if (waveSource != null)
            {
                waveSource.StopRecording();
                if (waveSource != null)
                {
                    waveSource.Dispose();
                    waveSource = null;
                }
                if (waveFile != null)
                {
                    waveFile.Dispose();
                    waveFile = null;
                }
            }
        }

        public void SetFileName(string fileName)
        {
            this.fileName = fileName;
        }

        private void waveSource_DataAvailable(object sender, WaveInEventArgs e)
        {
            if (waveFile != null)
            {
                if (e.Buffer != null && e.BytesRecorded > 0)
                {
                    voicebuff.Enqueue(e.Buffer);
                    //waveFile.Write(e.Buffer, 0, e.BytesRecorded);
                    waveFile.Flush();
                }

            }
        }

        public static byte[] Wavedata_Dequeue()
        {
            byte[] datas;
            voicebuff.TryDequeue(out datas);
            return datas;
        }

        private void waveSource_RecordingStopped(object sender, StoppedEventArgs e)
        {
            if (waveSource != null)
            {
                waveSource.Dispose();
                waveSource = null;
            }

            if (waveFile != null)
            {
                waveFile.Dispose();
                waveFile = null;
            }
        }
    }
}
