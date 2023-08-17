using Websocket.Client;
using System.Text.Json;
using System.Reactive.Linq;
using FunASRWSClient_Offline;

namespace WebSocketSpace
{
    internal class  CWebSocketClient
    {
        private static readonly Uri serverUri = new Uri($"ws://{WSClient_Offline.host}:{WSClient_Offline.port}"); // 你要连接的WebSocket服务器地址
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
                        recmessage(msg.Text);
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

        public async Task<Task> ClientSendFileFunc(string file_name)//文件转录
        {
            try
            {
                if (client.IsRunning)
                {
                    var exitEvent = new ManualResetEvent(false);
                    string path = Path.GetFileName(file_name);
                    string firstbuff = string.Format("{{\"mode\": \"offline\", \"wav_name\": \"{0}\", \"is_speaking\": true}}", Path.GetFileName(file_name));
                    client.Send(firstbuff);
                    showWAVForm(client, file_name);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            return Task.CompletedTask;
        }

        public void recmessage(string message)
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
            }
        }

        private void showWAVForm(WebsocketClient client, string file_name)
        {
            byte[] getbyte = FileToByte(file_name).Skip(44).ToArray();

            for (int i = 0; i < getbyte.Length; i += 1024000)
            {
                byte[] send = getbyte.Skip(i).Take(1024000).ToArray();
                client.Send(send);
                Thread.Sleep(5);
            }
            Thread.Sleep(10);
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