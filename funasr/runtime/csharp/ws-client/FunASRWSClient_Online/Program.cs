using AliFsmnVadSharp;
using NAudio.Wave;
using System.Collections.Concurrent;
using WebSocketSpace;
using NAudio.CoreAudioApi;
using System.IO;
using System.Collections.Specialized;

namespace FunASRWSClient_Online
{
    /// <summary>
    /// /主程序入口
    /// </summary>
    public class Program
    {
        private static void Main()
        {
            WSClient_Online m_funasrclient = new WSClient_Online();
            m_funasrclient.FunASR_Main();
        }
    }
    /// <summary>
    /// /主线程入口，初始化后读取数据
    /// </summary>
    public class WSClient_Online
    {
        /// <summary>
        /// FunASR客户端软件运行状态
        /// </summary>
        /// 
        public static string host = "0.0.0.0";
        public static string port = "10095";
        public static string onlineasrmode = string.Empty;
        private static WaveCollect m_wavecollect = new WaveCollect();
        private static CWebSocketClient m_websocketclient = new CWebSocketClient();
        public static readonly ConcurrentQueue<byte[]> ActiveAudioSet = new ConcurrentQueue<byte[]>();
        public static readonly ConcurrentQueue<string> AudioFileQueue = new ConcurrentQueue<string>();
        [STAThread]
        public void FunASR_Main()
        {
            loadconfig();
            //麦克风状态监测
            string errorStatus = string.Empty;
            if (GetCurrentMicVolume() == -2)
                errorStatus = "注意：麦克风被设置为静音！";
            else if (GetCurrentMicVolume() == -1)
                errorStatus = "注意：麦克风未连接！";
            else if (GetCurrentMicVolume() == 0)
                errorStatus = "注意：麦克风声音设置为0！";

            //初始化通信连接
            string commstatus = ClientConnTest();
            if (commstatus != "通信连接成功")
                errorStatus = commstatus;
            //程序初始监测异常--报错、退出
            if (errorStatus != string.Empty)
            {
                Environment.Exit(0);//报错方式待加
            }

            //启动客户端向服务端发送音频数据线程
            Thread SendAudioThread = new Thread(SendAudioToSeverAsync);
            SendAudioThread.Start();

            //启动音频文件转录线程
            Thread AudioFileThread = new Thread(SendAudioFileToSeverAsync);
            AudioFileThread.Start();

            while (true)
            {
                Console.WriteLine("请选择语音识别方式：1.离线文件转写；2.实时语音识别");
                string str = Console.ReadLine();
                if (str != string.Empty)
                {
                    if (str == "1")//离线文件转写
                    {
                        onlineasrmode = "offline";
                        Console.WriteLine("请输入转录文件路径");
                        str = Console.ReadLine();
                        if (!string.IsNullOrEmpty(str))
                            AudioFileQueue.Enqueue(str);
                    }
                    else if (str == "2")//实时语音识别
                    {
                        Console.WriteLine("请输入实时语音识别模式：1.online；2.2pass");
                        str = Console.ReadLine();
                        OnlineASR(str);
                    }
                }
            }
        }
        private void loadconfig()
        {
            string filePath = "config.ini";
            NameValueCollection settings = new NameValueCollection();
            using (StreamReader reader = new StreamReader(filePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    // 忽略空行和注释
                    if (string.IsNullOrEmpty(line) || line.StartsWith(";") || line.StartsWith("#"))
                        continue;
                    // 解析键值对
                    int equalsIndex = line.IndexOf('=');
                    if (equalsIndex > 0)
                    {
                        string key = line.Substring(0, equalsIndex).Trim();
                        string value = line.Substring(equalsIndex + 1).Trim();
                        if (key == "host")
                            host = value;
                        else if (key == "port")
                            port = value;
                    }
                }
            }
        }
        private void OnlineASR(string str)
        {
            if (!string.IsNullOrEmpty(str))
            {
                if (str == "1")//实时语音识别
                    onlineasrmode = "online";
                else if (str == "2")//实时语音识别-动态修正
                    onlineasrmode = "2pass";
            }
            //开始录制声音、发送识别
            if (onlineasrmode != string.Empty)
            {
                m_wavecollect.StartRec();
                m_websocketclient.ClientFirstConnOnline(onlineasrmode);
                try
                {
                    while (true)
                    {
                        if (!WaveCollect.voicebuff.IsEmpty)
                        {
                            byte[] buff;
                            int buffcnt = WaveCollect.voicebuff.Count;
                            WaveCollect.voicebuff.TryDequeue(out buff);
                            if (buff != null)
                                ActiveAudioSet.Enqueue(buff);
                        }
                        else
                        {
                            if (Console.KeyAvailable)
                            {
                                var key = Console.ReadKey(true);

                                // 检测到按下Ctrl+C
                                if ((key.Modifiers & ConsoleModifiers.Control) != 0 && key.Key == ConsoleKey.C)
                                {
                                    // 执行相应的操作
                                    Console.WriteLine("Ctrl+C Pressed!");
                                    // 退出循环或执行其他操作
                                    break;
                                }
                            }
                            else
                            {
                                Thread.Sleep(10);
                            }
                        }
                    }
                }
                catch
                {
                    Console.WriteLine("实时识别出现异常！");
                }
                finally
                {
                    m_wavecollect.StopRec();
                    m_websocketclient.ClientLastConnOnline();
                }
            }
        }

        private string ClientConnTest()
        {
            //WebSocket连接状态监测
            Task<string> websocketstatus = m_websocketclient.ClientConnTest();
            if (websocketstatus != null && websocketstatus.Result.IndexOf("成功") == -1)
                return websocketstatus.Result;
            return "通信连接成功";
        }
        private void SendAudioFileToSeverAsync()
        {
            while (true)
            {
                Thread.Sleep(1000);
                if (AudioFileQueue.Count > 0)
                {
                    string filepath = string.Empty;
                    AudioFileQueue.TryDequeue(out filepath);
                    if (filepath != string.Empty && filepath != null)
                    {
                        m_websocketclient.ClientSendFileFunc(filepath);
                    }
                }
                else
                {
                    Thread.Sleep(100);
                }
            }
        }
        private void SendAudioToSeverAsync()
        {
            while (true)
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
                        m_websocketclient.ClientSendAudioFunc(mArray);
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