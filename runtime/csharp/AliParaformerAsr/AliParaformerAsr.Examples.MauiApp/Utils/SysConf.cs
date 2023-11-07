using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MauiApp1.Utils
{
    internal class SysConf
    {
#if WINDOWS
    private static string _applicationBase = Microsoft.Maui.Storage.FileSystem.AppDataDirectory;// "/data/user/0/com.companyname.mauiapp1/files/AllModels";//"/data/data/com.companyname.mauiapp1/files/Assets";// AppDomain.CurrentDomain.BaseDirectory;//
#else
        private static string _applicationBase = AppDomain.CurrentDomain.BaseDirectory;
#endif
        public SysConf() { }

        public static string ApplicationBase { get => _applicationBase; set => _applicationBase = value; }
    }
}
