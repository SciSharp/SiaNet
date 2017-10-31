using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Common
{
    public delegate void WriteLog(string message);

    public class Logging
    {
        public static event WriteLog OnWriteLog;

        public static void WriteTrace(string message)
        {
            OnWriteLog(message);
        }

        public static void WriteTrace(Exception ex)
        {
            OnWriteLog("Exception: " + ex.Message);
        }
    }
}
