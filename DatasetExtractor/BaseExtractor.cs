using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace DatasetExtractor
{
    internal class BaseExtractor
    {
        int downloadPercentPrev = 0;

        public void DownloadFile(string serverPath, string localPath)
        {
            if(File.Exists(localPath))
            {
                Console.WriteLine("File exist: " + localPath);
                return;
            }

            Console.WriteLine("Downloading file: " + serverPath);
            downloadPercentPrev = 0;
            WebClient wb = new WebClient();
            wb.DownloadProgressChanged += Wb_DownloadProgressChanged;
            wb.DownloadFileTaskAsync(new Uri(serverPath), localPath).Wait();
            Console.WriteLine("Download Complete");
        }

        /// <summary>
        /// Handles the DownloadProgressChanged event of the Wb control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="DownloadProgressChangedEventArgs"/> instance containing the event data.</param>
        public void Wb_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            if (e.ProgressPercentage == downloadPercentPrev)
                return;

            downloadPercentPrev = e.ProgressPercentage;
            if (e.ProgressPercentage % 5 == 0)
                Console.WriteLine(string.Format("Download Progress: {0}%", e.ProgressPercentage));
        }
    }
}
