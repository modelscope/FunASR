using System.Collections.Specialized;
using WebSocketSpace;

namespace FunASRWSClient_Offline
{
    /// <summary>
    /// /主程序入口
    /// </summary>
    public class Program
    {
        private static void Main()
        {
            WSClient_Offline m_funasrclient = new WSClient_Offline();
            m_funasrclient.FunASR_Main();
        }
    }

    public class WSClient_Offline
    {
        public static string host = "0.0.0.0";
        public static string port = "10095";
        private static CWebSocketClient m_websocketclient = new CWebSocketClient();
        [STAThread]
        public async void FunASR_Main()
        {
            loadconfig();
            //初始化通信连接
            string errorStatus = string.Empty;
            string commstatus = ClientConnTest();
            if (commstatus != "通信连接成功")
                errorStatus = commstatus;
            //程序初始监测异常--报错、退出
            if (errorStatus != string.Empty)
            {
                //报错方式待加
                Environment.Exit(0);
            }

            //循环输入推理文件
            while (true)
            {
                Console.WriteLine("请输入转录文件路径：");
                string filepath = Console.ReadLine();
                if (filepath != string.Empty && filepath != null)
                {
                     await m_websocketclient.ClientSendFileFunc(filepath);
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
        private static string ClientConnTest()
        {
            //WebSocket连接状态监测
            Task<string> websocketstatus = m_websocketclient.ClientConnTest();
            if (websocketstatus != null && websocketstatus.Result.IndexOf("成功") == -1)
                return websocketstatus.Result;
            return "通信连接成功";
        }
    }
}