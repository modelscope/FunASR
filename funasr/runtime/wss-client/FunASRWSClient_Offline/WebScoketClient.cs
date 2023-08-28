using Websocket.Client;
using System.Text.Json;
using System.Reactive.Linq;
using FunASRWSClient_Offline;
using System.Text.RegularExpressions;

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
                    string firstbuff = string.Format("{{\"mode\": \"offline\", \"wav_name\": \"{0}\", \"is_speaking\": true,\"hotwords\":\"{1}\"}}", Path.GetFileName(file_name), WSClient_Offline.hotword);
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
                    string timestamp = string.Empty;
                    JsonDocument jsonDoc = JsonDocument.Parse(message);
                    JsonElement root = jsonDoc.RootElement;
                    string mode = root.GetProperty("mode").GetString();
                    string text = root.GetProperty("text").GetString(); 
                    string name = root.GetProperty("wav_name").GetString();
                    if (message.IndexOf("timestamp") != -1)
                    {
                        Console.WriteLine($"文件名称:{name}");
                        //识别内容处理
                        text = text.Replace(",", "。");
                        text = text.Replace("?", "。");
                        List<string> sens = text.Split("。").ToList();
                        //时间戳处理
                        timestamp = root.GetProperty("timestamp").GetString();
                        List<List<int>> data = new List<List<int>>();
                        string pattern = @"\[(\d+),(\d+)\]";
                        foreach (Match match in Regex.Matches(timestamp, pattern))
                        {
                            int start = int.Parse(match.Groups[1].Value);
                            int end = int.Parse(match.Groups[2].Value);
                            data.Add(new List<int> { start, end });
                        }
                        int count = 0;
                        for (int i = 0; i< sens.Count;  i++)
                        {
                            if (sens[i].Length == 0)
                                continue;
                            Console.WriteLine(string.Format($"[{data[count][0]}-{data[count + sens[i].Length - 1][1]}]:{sens[i]}"));
                            count += sens[i].Length;
                        }
                    }
                    else
                    {
                        Console.WriteLine($"文件名称:{name} 文件转录内容: {text} 时间戳：{timestamp}");
                    }
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