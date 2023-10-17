using System.Net.WebSockets;
using Websocket.Client;
using System.Text.Json;
using NAudio.Wave;
using AliFsmnVadSharp;
using System.Reactive.Linq;
using FunASRWSClient_Online;

namespace WebSocketSpace
{
    internal class  CWebSocketClient
    {
        private static int chunk_interval = 10;
        private static int[] chunk_size = new int[] { 5, 10, 5 };
        private static readonly Uri serverUri = new Uri($"ws://{WSClient_Online.host}:{WSClient_Online.port}"); // 你要连接的WebSocket服务器地址
        private static WebsocketClient client = new WebsocketClient(serverUri);
        public async Task<string> ClientConnTest()
        {
            string commstatus = "WebSocket通信连接失败";
            try
            {
                client.Name = "funasr";
                client.ReconnectTimeout = null;
                client.ReconnectionHappened.Subscribe(info =>
                   Console.WriteLine($"Reconnection happened, type: {info.Type}, url: {client.Url}"));
                client.DisconnectionHappened.Subscribe(info =>
                    Console.WriteLine($"Disconnection happened, type: {info.Type}"));

                client
                    .MessageReceived
                    .Where(msg => msg.Text != null)
                    .Subscribe(msg =>
                {
                    rec_message(msg.Text, client);
                });

                await client.Start();

                if (client.IsRunning)
                    commstatus = "WebSocket通信连接成功";
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                client.Dispose();
            }

            return commstatus;
        }

        public bool ClientFirstConnOnline(string asrmode)
        {
            if (client.IsRunning)
            {
                string firstbuff = string.Format("{{\"mode\": \"{0}\", \"chunk_size\": [{1},{2},{3}], \"chunk_interval\": {4}, \"wav_name\": \"microphone\", \"is_speaking\": true}}"
                       , asrmode, chunk_size[0], chunk_size[1], chunk_size[2], chunk_interval);
                Task.Run(() => client.Send(firstbuff));
            }
            else
            {
                client.Reconnect();
                return false;
            }

            return true;
        }
        public bool ClientSendAudioFunc(byte[] buff)    //实时识别
        {
            if (client.IsRunning)
            {
                ////发送音频数据
                int CHUNK = WaveCollect.wave_buffer_collectfrequency / 1000 * 60 * chunk_size[1] / chunk_interval;
                for (int i = 0; i < buff.Length; i += CHUNK)
                {
                    byte[] send = buff.Skip(i).Take(CHUNK).ToArray();
                    Task.Run(() => client.Send(send));
                    Thread.Sleep(1);
                }
            }
            else
            {
                client.Reconnect();
                return false;
            }

            return true;
        }
        public void ClientLastConnOnline()
        {
            Task.Run(() => client.Send("{\"is_speaking\": false}"));
        }

        public int ClientSendFileFunc(string file_name)//文件转录 0:发送成功 ret -1:文件类型不支持 -2:通信断开
        {
            string fileExtension = Path.GetExtension(file_name);
            fileExtension = fileExtension.Replace(".", "");
            if (!(fileExtension == "mp3" || fileExtension == "mp4" || fileExtension == "wav" || fileExtension == "pcm"))
                return -1;

            if (client.IsRunning)
            {
                if (fileExtension == "wav" || fileExtension == "pcm")
                {
                    string firstbuff = string.Format("{{\"mode\": \"office\", \"chunk_size\": [{0},{1},{2}], \"chunk_interval\": {3}, \"wav_name\": \"{4}\", \"is_speaking\": true, \"wav_format\":\"pcm\"}}"
                        , chunk_size[0], chunk_size[1], chunk_size[2], chunk_interval, Path.GetFileName(file_name));
                    Task.Run(() => client.Send(firstbuff));
                    if (fileExtension == "wav")
                        showWAVForm(file_name);
                    else if (fileExtension == "pcm")
                        showWAVForm_All(file_name);
                }
                else if (fileExtension == "mp3" || fileExtension == "mp4")
                {
                    string firstbuff = string.Format("{{\"mode\": \"offline\", \"chunk_size\": \"{0},{1},{2}\", \"chunk_interval\": {3}, \"wav_name\": \"{4}\", \"is_speaking\": true, \"wav_format\":\"{5}\"}}"
                        , chunk_size[0], chunk_size[1], chunk_size[2], chunk_interval, Path.GetFileName(file_name), fileExtension);
                    Task.Run(() => client.Send(firstbuff));
                    showWAVForm_All(file_name);
                }
            }
            else
            {
                client.Reconnect();
                return -2;
            }

            return 0;
        }
        private string recbuff = string.Empty;//接收累计缓存内容
        private string onlinebuff = string.Empty;//接收累计在线缓存内容
        public void rec_message(string message, WebsocketClient client)
        {
            if (message != null)
            {
                try
                {
                    string name = string.Empty;
                    JsonDocument jsonDoc = JsonDocument.Parse(message);
                    JsonElement root = jsonDoc.RootElement;
                    string mode = root.GetProperty("mode").GetString();
                    string text = root.GetProperty("text").GetString();
                    bool isfinal = root.GetProperty("is_final").GetBoolean();
                    if (message.IndexOf("wav_name  ") != -1)
                        name = root.GetProperty("wav_name").GetString();

                    //if (name == "microphone")
                    //    Console.WriteLine($"实时识别内容: {text}");
                    //else
                    //    Console.WriteLine($"文件名称:{name} 文件转录内容: {text}");

                    if (mode == "2pass-online" && WSClient_Online.onlineasrmode != "offline")
                    {
                        onlinebuff += text;
                        Console.WriteLine(recbuff + onlinebuff);
                    }
                    else if (mode == "2pass-offline")
                    {
                        recbuff += text;
                        onlinebuff = string.Empty;
                        Console.WriteLine(recbuff);
                    }

                    if (isfinal && WSClient_Online.onlineasrmode != "offline")//未结束当前识别
                    {
                        recbuff = string.Empty;
                    }
                }
                catch (JsonException ex)
                {
                    Console.WriteLine("JSON 解析错误: " + ex.Message);
                }
            }
        }

        private void showWAVForm(string file_name)
        {
            byte[] getbyte = FileToByte(file_name).Skip(44).ToArray();

            for (int i = 0; i < getbyte.Length; i += 102400)
            {
                byte[] send = getbyte.Skip(i).Take(102400).ToArray();
                Task.Run(() => client.Send(send));
                Thread.Sleep(5);
            }
            Thread.Sleep(100);
            Task.Run(() => client.Send("{\"is_speaking\": false}"));
        }

        private void showWAVForm_All(string file_name)
        {
            byte[] getbyte = FileToByte(file_name).ToArray();

            for (int i = 0; i < getbyte.Length; i += 1024000)
            {
                byte[] send = getbyte.Skip(i).Take(1024000).ToArray();
                Task.Run(() => client.Send(send));
                Thread.Sleep(5);
            }
            Thread.Sleep(10);
            Task.Run(() => client.Send("{\"is_speaking\": false}"));
        }

        public byte[] FileToByte(string fileUrl)
        {
            try
            {
                using (FileStream fs = new FileStream(fileUrl, FileMode.Open, FileAccess.Read))
                {
                    byte[] byteArray = new byte[fs.Length];
                    fs.Read(byteArray, 0, byteArray.Length);
                    return byteArray;
                }
            }
            catch
            {
                return null;
            }
        }
    }
}