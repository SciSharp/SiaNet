using SiaNet;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace SieNet.Examples
{
    public class DatasetInfo
    {
        public string BaseFolder { get; set; }

        public string Train { get; set; }

        public string Test { get; set; }
    }

    public class Downloader
    {
        static string serverUrl = "https://sianet.blob.core.windows.net/dataset/{0}.zip";

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

        }

        private static void CheckAndDownload(SampleDataset datasetName, string fileName, bool force = false)
        {
            DatasetInfo datasetInfo = GetSamplePath(datasetName);
            if(force)
            {
                Directory.Delete(datasetInfo.BaseFolder);
            }

            if(!Directory.Exists(datasetInfo.BaseFolder))
            {
                Directory.CreateDirectory(datasetInfo.BaseFolder);
            }

            string localfile = string.Format("{0}\\data.zip", datasetInfo.BaseFolder);
            if (!File.Exists(localfile))
            {
                DownloadFile(string.Format(serverUrl, fileName), localfile);
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
            WebClient wb = new WebClient();
            wb.DownloadFile(serverPath, localPath);
        }
    }
}
