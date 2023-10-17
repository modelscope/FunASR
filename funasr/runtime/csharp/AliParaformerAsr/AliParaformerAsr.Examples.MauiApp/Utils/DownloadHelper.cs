using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Security.Cryptography;

namespace MauiApp1.Utils
{
    public enum DownloadState
    {
        cancelled = 0, // 取消
        inprogres = 1, //进行中
        completed = 2, //完成
        error = 3, //错误
        existed = 4,//已存在
        noexisted = 5,//已存在

    }
    public delegate void DelegateDone(int progress, DownloadState downloadState,string filename, string msg = "");
    internal class DownloadHelper : INotifyPropertyChanged
    {
        private DelegateDone _callback;
        private bool _isDownloading = false;
        private int _progress;
        private string _fileName;
        private object _lockobj = new object();
        private WebClient _webClient;
        private ICommand DownloadCommand { get; set; }
        public bool IsDownloading
        {
            get => _isDownloading;
            set
            {
                if (value != _isDownloading)
                {
                    _isDownloading = value;
                    NotifyPropertyChanged("IsDownloading");
                }
            }
        }

        public string FileName
        {
            get => _fileName;
            set
            {
                if (value != _fileName)
                {
                    _fileName = value;
                    NotifyPropertyChanged("FileName");
                }
            }
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        protected void NotifyPropertyChanged(string property)
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(property));
            }
        }

        public DownloadHelper()
        {
        }


        public DownloadHelper(DelegateDone callback)
        {
            _callback = callback;
        }

        public void DownloadCreate(string downloadUrl, string fileName, string rootFolderName, string subFolderName)
        {
            lock (_lockobj)
            {
                //IsDownloading = true;
                FileName = fileName;
                var downloadFolder = Path.Combine(SysConf.ApplicationBase, rootFolderName);
                var modelFolder = Path.Combine(downloadFolder, subFolderName);
                Directory.CreateDirectory(downloadFolder);
                Directory.CreateDirectory(modelFolder);
                string fileFullname = Path.Combine(modelFolder, fileName);
                if (!File.Exists(fileFullname))
                {
                    _webClient = new WebClient();
                    _webClient.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36");
                    _webClient.DownloadFileCompleted += OnDownloadCompleted;
                    _webClient.DownloadProgressChanged += OnDownloadProgressChanged;
                    DownloadCommand = new Command(() =>
                    {
                        if (!IsDownloading)
                        {
                            IsDownloading = true;
                            _webClient.DownloadFileAsync(new Uri(downloadUrl), fileFullname);
                        }
                    });
                }
                else
                {
                    DownloadCommand = null;
                    if (_callback != null)
                    {
                        _callback(100, DownloadState.existed, FileName);
                    }
                }
            }
        }

        public void DownloadStart()
        {
            if (DownloadCommand != null && DownloadCommand.CanExecute(null))
            {
                DownloadCommand.Execute(null);
            }
        }

        public void DownloadCancel()
        {
            if (_webClient != null)
            {
                _webClient.CancelAsync();
            }
        }

        public bool GetDownloadState(string[] fileNames, string rootFolderName, string subFolderName, string[] md5Strs)
        {
            bool state=true;
            var downloadFolder = Path.Combine(SysConf.ApplicationBase, rootFolderName);
            var modelFolder = Path.Combine(downloadFolder, subFolderName);
            Directory.CreateDirectory(downloadFolder);
            Directory.CreateDirectory(modelFolder);
            for (int i = 0; i < fileNames.Length; i++)
            {
                string fileFullname = Path.Combine(modelFolder, fileNames[i]);
                if (File.Exists(fileFullname))
                {                    
                    FileInfo fileInfo = new FileInfo(fileFullname);
                    if (fileInfo.Length == 0)
                    {
                        state = state && false;
                    }
                    else
                    {
                        string md5str = GetMD5Hash(fileFullname);
                        if (!md5str.Equals(md5Strs[i]))
                        {
                            state = state && false;
                        }
                    }
                }
                else
                {
                    state = state && false;
                }
            }
            
            return state;
        }

        public void DownloadCheck(string fileName, string rootFolderName, string subFolderName,string md5Str)
        {
            var downloadFolder = Path.Combine(SysConf.ApplicationBase, rootFolderName);
            var modelFolder = Path.Combine(downloadFolder, subFolderName);
            Directory.CreateDirectory(downloadFolder);
            Directory.CreateDirectory(modelFolder);
            string fileFullname = Path.Combine(modelFolder, fileName);
            if (File.Exists(fileFullname))
            {
                FileInfo fileInfo = new FileInfo(fileFullname);
                if (fileInfo.Length == 0)
                {
                    File.Delete(fileFullname);
                    string filename = fileInfo.Name;
                    _callback(0, DownloadState.noexisted, filename);
                }
                else
                {
                    string md5str = GetMD5Hash(fileFullname);
                    if (!md5str.Equals(md5Str))
                    {
                        File.Delete(fileFullname);
                        string filename = fileInfo.Name;
                        _callback(0, DownloadState.noexisted, filename);
                    }
                }
            }
            else
            {
                string filename = fileName;
                _callback(0, DownloadState.noexisted, filename);
            }
        }


        private void OnDownloadProgressChanged(object? sender, DownloadProgressChangedEventArgs e)
        {
            string filename = FileName;
            _progress = (int)(e.BytesReceived * 100 / e.TotalBytesToReceive);
            if (_callback != null)
            {
                _callback(_progress, DownloadState.inprogres, filename);
            }
        }
        private void OnDownloadCompleted(object? sender, AsyncCompletedEventArgs e)
        {
            string filename = FileName;
            IsDownloading = false;
            if (_callback != null)
            {
                if (e.Cancelled)
                {
                    _callback(_progress, DownloadState.cancelled, filename);
                }
                else if (e.Error != null)
                {
                    _callback(_progress, DownloadState.error,filename, msg: e.Error.Message);
                }
                else
                {
                    _callback(_progress, DownloadState.completed, filename);
                }
            }
        }

        public string GetMD5Hash(string path)
        {
            using (FileStream fs = new FileStream(path, FileMode.Open))
            {
                StringBuilder sb = new StringBuilder();
                MD5 md5 = MD5.Create();
                byte[] array = md5.ComputeHash(fs);
                fs.Close();
                for (int i = 0; i < array.Length; i++)
                {
                    sb.Append(array[i].ToString("x2"));
                }
                return sb.ToString();
            }
        }
    }
}
