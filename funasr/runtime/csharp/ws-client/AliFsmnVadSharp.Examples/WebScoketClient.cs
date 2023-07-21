using System;
using System.Linq;
using System.Net.Security;
using System.Net.Sockets;
using System.Net.WebSockets;
using System.Security.Authentication;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Websocket.Client;
using System.Text.Json;
using System.Net;
using NAudio.Wave;
using System.ComponentModel.Design;

namespace WebSocketSpace
{
    internal class  CWebSocketClient
    {
        private static readonly Uri serverUri = new Uri("ws://172.21.1.27:10095"); // 你要连接的WebSocket服务器地址
        private static WebsocketClient client = new WebsocketClient(serverUri);

        public async Task<string> ClientConnTest()
        {
            string commstatus = "WebSocket通信连接失败";
            try
            {
                using (var client = new WebsocketClient(serverUri))
                {
                    client.ReconnectTimeout = TimeSpan.FromSeconds(5);
                    await client.StartOrFail();
                    if (client.IsRunning)
                    {
                        commstatus = "WebSocket通信连接成功";
                        client.NativeClient.Abort();
                    }
                }
            }
            catch (Exception ex) 
            {
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                client.Dispose();
            }
            return commstatus;
        }

        public async Task<Task> ClientSendAudioFunc(byte[] buff)    //实时识别
        {
            var exitEvent = new ManualResetEvent(false);
            using (var client = new WebsocketClient(serverUri))
            {
                client.ReconnectTimeout = TimeSpan.FromSeconds(5);
                await client.StartOrFail();
                bool status = client.IsRunning;
                client.Send("{\"mode\": \"offline\", \"wav_name\": \"asr_stream\", \"is_speaking\": true}");
                for (int i = 0; i < buff.Length; i += 1024)
                {
                    byte[] send = buff.Skip(i).Take(1024).ToArray();
                    client.Send(send);
                }
                Thread.Sleep(100);
                client.Send("{\"is_speaking\": false}");
                client.MessageReceived.Subscribe(msg => recmessage(msg.Text, client, exitEvent));
                exitEvent.WaitOne();
            }
            return Task.CompletedTask;
        }

        public async Task<Task> ClientSendFileFunc(string file_name)//文件转录
        {
            var exitEvent = new ManualResetEvent(false);
            using (var client = new WebsocketClient(serverUri))
            {
                client.ReconnectTimeout = TimeSpan.FromSeconds(30);
                await client.StartOrFail();
                string path = Path.GetFileName(file_name);
                string firstbuff = string.Format("{{\"mode\": \"offline\", \"wav_name\": \"{0}\", \"is_speaking\": true}}", Path.GetFileName(file_name));
                client.Send(firstbuff);
                showWAVForm(client, file_name);
                client.MessageReceived.Subscribe(msg => recmessage(msg.Text, client, exitEvent));
                exitEvent.WaitOne();
            }
            return Task.CompletedTask;
        }

        public void recmessage(string message, WebsocketClient client, ManualResetEvent exitEvent)
        {
            if (message != null)
            {
                try
                {
                    JsonDocument jsonDoc = JsonDocument.Parse(message);
                    JsonElement root = jsonDoc.RootElement;
                    string mode = root.GetProperty("mode").GetString();
                    string text = root.GetProperty("text").GetString();
                    string name = root.GetProperty("wav_name").GetString();
                    if(name == "asr_stream")
                        Console.WriteLine($"实时识别内容: {text}");
                    else
                        Console.WriteLine($"文件名称:{name} 文件转录内容: {text}");
                }
                catch (JsonException ex)
                {
                    Console.WriteLine("JSON 解析错误: " + ex.Message);
                }
                finally
                {
                    exitEvent.Set();
                    client.NativeClient.Abort();
                    client.Dispose();
                }
            }
        }

        private void showWAVForm(WebsocketClient client, string file_name)
        {
            byte[] getbyte = FileToByte(file_name).Skip(44).ToArray();

            for (int i = 0; i < getbyte.Length; i += 1024)
            {
                byte[] send = getbyte.Skip(i).Take(1024).ToArray();
                client.Send(send);
            }
            Thread.Sleep(100);
            client.Send("{\"is_speaking\": false}");
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