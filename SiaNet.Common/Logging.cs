using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Common
{
    /// <summary>
    /// Delegate for Write Log event
    /// </summary>
    /// <param name="message">The message.</param>
    public delegate void WriteLog(string message);

    /// <summary>
    /// Logging class for writing logs
    /// </summary>
    public class Logging
    {
        public static event WriteLog OnWriteLog;

        /// <summary>
        /// Writes the trace with message
        /// </summary>
        /// <param name="message">The message.</param>
        public static void WriteTrace(string message)
        {
            OnWriteLog(message);
        }

        /// <summary>
        /// Writes the trace with exception object
        /// </summary>
        /// <param name="ex">The ex.</param>
        public static void WriteTrace(Exception ex)
        {
            OnWriteLog("Exception: " + ex.Message);
        }
    }
}
