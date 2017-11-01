using SiaNet;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Common
{
    public class DatasetInfo
    {
        public string BaseFolder { get; set; }

        public string Train { get; set; }

        public string Test { get; set; }
    }

    public class Downloader
    {
        static string serverUrl = "https://sianet.blob.core.windows.net/dataset/{0}";
        static int downloadPercentPrev = 0;
        public static void DownloadSample(SampleDataset datasetName, bool force = false)
        {
            string filename = "";
            switch (datasetName)
            {
                case SampleDataset.HousingRegression:
                    filename = "housing.zip";
                    break;
                case SampleDataset.MNIST:
                    filename = "MNIST.zip";
                    break;
                case SampleDataset.Cifar10:
                    filename = "CIFAR-10.zip";
                    break;
                case SampleDataset.Cifar100:
                    filename = "CIFAR-100.zip";
                    break;
                case SampleDataset.Flowers:
                    filename = "Flowers.zip";
                    break;
                case SampleDataset.Grocery:
                    filename = "Grocery.zip";
                    break;
                default:
                    break;
            }

            CheckAndDownload(datasetName, filename, force);
        }

        public static DatasetInfo GetSamplePath(SampleDataset datasetName)
        {
            DatasetInfo path = new DatasetInfo();
            string baseFolder = string.Format("{0}\\SiaNet\\dataset", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
            string dataFolder = "";
            switch (datasetName)
            {
                case SampleDataset.HousingRegression:
                    dataFolder = string.Format("{0}\\housing", baseFolder);
                    path.Train = string.Format("{0}\\train.csv", dataFolder);
                    path.Test = string.Format("{0}\\test.csv", dataFolder);
                    path.BaseFolder = dataFolder;
                    break;
                case SampleDataset.MNIST:
                    dataFolder = string.Format("{0}\\MNIST", baseFolder);
                    path.Train = string.Format("{0}\\train.txt", dataFolder);
                    path.Test = string.Format("{0}\\test.txt", dataFolder);
                    path.BaseFolder = dataFolder;
                    break;
                case SampleDataset.Cifar10:
                    dataFolder = string.Format("{0}\\Cifar10", baseFolder);
                    path.Train = string.Format("{0}\\train.txt", dataFolder);
                    path.Test = string.Format("{0}\\test.txt", dataFolder);
                    path.BaseFolder = dataFolder;
                    break;
                case SampleDataset.Cifar100:
                    dataFolder = string.Format("{0}\\Cifar100", baseFolder);
                    path.Train = string.Format("{0}\\train.txt", dataFolder);
                    path.Test = string.Format("{0}\\test.txt", dataFolder);
                    path.BaseFolder = dataFolder;
                    break;
                case SampleDataset.Flowers:
                    dataFolder = string.Format("{0}\\Flowers", baseFolder);
                    path.Train = string.Format("{0}\\6k_img_map.txt", dataFolder);
                    path.Test = string.Format("{0}\\val_map.txt", dataFolder);
                    path.BaseFolder = dataFolder;
                    break;
                case SampleDataset.Grocery:
                    dataFolder = string.Format("{0}\\Grocery", baseFolder);
                    path.Train = string.Format("{0}\\train.txt", dataFolder);
                    path.Test = string.Format("{0}\\test.txt", dataFolder);
                    path.BaseFolder = dataFolder;
                    break;
                default:
                    break;
            }

            return path;
        }

        public static void DownloadModel(string modelPath)
        {
            string baseFolder = string.Format("{0}\\SiaNet\\models", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
            if (!Directory.Exists(baseFolder))
            {
                Directory.CreateDirectory(baseFolder);
            }

            string filename = Path.GetFileName(modelPath);
            string fullpath = baseFolder + "\\" + filename;

            if(File.Exists(fullpath))
            {
                return;
            }

            Logging.WriteTrace("Downloading model: " + modelPath);
            DownloadFile(modelPath, fullpath);
            Logging.WriteTrace("Download complete");
        }

        private static void CheckAndDownload(SampleDataset datasetName, string fileName, bool force = false)
        {
            DatasetInfo datasetInfo = GetSamplePath(datasetName);
            if (force)
            {
                Directory.Delete(datasetInfo.BaseFolder);
            }
            else
            {
                if (Directory.Exists(datasetInfo.BaseFolder))
                {
                    return;
                }
            }

            if(!Directory.Exists(datasetInfo.BaseFolder))
            {
                Directory.CreateDirectory(datasetInfo.BaseFolder);
            }

            string localfile = string.Format("{0}\\data.zip", datasetInfo.BaseFolder);
            if (!File.Exists(localfile))
            {
                Logging.WriteTrace("Downloading sample data: " + fileName);
                DownloadFile(string.Format(serverUrl, fileName), localfile);
                Logging.WriteTrace("Download complete");
            }

            UnzipFile(datasetInfo.BaseFolder, "data.zip");
        }

        private static void UnzipFile(string baseFolder, string filename)
        {
            string zipfile = string.Format("{0}\\{1}", baseFolder, filename);
            
            System.IO.Compression.ZipFile.ExtractToDirectory(zipfile, baseFolder);
            File.Delete(zipfile);
        }

        private static void DownloadFile(string serverPath, string localPath)
        {
            downloadPercentPrev = 0;
            WebClient wb = new WebClient();
            wb.DownloadProgressChanged += Wb_DownloadProgressChanged;
            wb.DownloadFileTaskAsync(new Uri(serverPath), localPath).Wait();
        }

        private static void Wb_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            if (e.ProgressPercentage == downloadPercentPrev)
                return;

            downloadPercentPrev = e.ProgressPercentage;
            if(e.ProgressPercentage % 5 == 0)
                Logging.WriteTrace(string.Format("Download Progress: {0}%", e.ProgressPercentage));
        }
    }
}
