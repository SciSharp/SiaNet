using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace SieNet.Examples
{
    class SampleDownloader
    {
        static string housingRegressionDataUrl = "https://siastore.blob.core.windows.net/demo/data/housing/{0}";
        static string MNISTDataUrl = "https://siastore.blob.core.windows.net/demo/data/housing/{0}";

        public static void CheckAndDownloadHousingRegression()
        {
            string dataFolder = string.Format("{0}\\SiaNet\\housing", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
            if(!Directory.Exists(dataFolder))
            {
                Directory.CreateDirectory(dataFolder);
            }

            string localfile = string.Format("{0}\\train.csv", dataFolder);
            if (!System.IO.File.Exists(localfile))
            {
                DownloadFile(string.Format(housingRegressionDataUrl, "train.csv"), localfile);
            }

            localfile = string.Format("{0}\\test.csv", dataFolder);
            if (!System.IO.File.Exists(localfile))
            {
                DownloadFile(string.Format(housingRegressionDataUrl, "test.csv"), localfile);
            }
        }

        public static void CheckAndDownloadMNIST()
        {
            string dataFolder = string.Format("{0}\\SiaNet\\MNIST", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));

            if (!Directory.Exists(dataFolder))
            {
                Directory.CreateDirectory(dataFolder);
            }

            string localfile = string.Format("{0}\\train.txt", dataFolder);
            if (!System.IO.File.Exists(localfile))
            {
                DownloadFile(string.Format(MNISTDataUrl, "train.txt"), localfile);
            }

            localfile = string.Format("{0}\\test.txt", dataFolder);
            if (!System.IO.File.Exists(localfile))
            {
                DownloadFile(string.Format(MNISTDataUrl, "test.txt"), localfile);
            }
        }

        private static void DownloadFile(string serverPath, string localPath)
        {
            WebClient wb = new WebClient();
            wb.DownloadFile(serverPath, localPath);
        }
    }
}
