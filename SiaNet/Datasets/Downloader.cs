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
    public class DatasetPath
    {
        public string Train { get; set; }

        public string Test { get; set; }
    }

    public class SampleDownloader
    {
        static string housingRegressionDataUrl = "https://siastore.blob.core.windows.net/demo/data/housing/{0}";
        static string MNISTDataUrl = "https://siastore.blob.core.windows.net/demo/data/MNIST/{0}";
        static string cifar10DataUrl = "https://siastore.blob.core.windows.net/demo/data/Cifar10/{0}";
        static string cifar100DataUrl = "https://siastore.blob.core.windows.net/demo/data/Cifar100/{0}";


        public void DownloadSample(SampleDataset datasetName, bool force = false)
        {
            switch (datasetName)
            {
                case SampleDataset.HousingRegression:
                    CheckAndDownloadHousingRegression(force);
                    break;
                case SampleDataset.MNIST:
                    CheckAndDownloadMNIST(force);
                    break;
                case SampleDataset.Cifar10:
                    CheckAndDownloadCifar10(force);
                    break;
                case SampleDataset.Cifar100:
                    break;
                default:
                    break;
            }
        }

        public DatasetPath GetSamplePath(SampleDataset datasetName)
        {
            DatasetPath path = new DatasetPath();
            string dataFolder = "";
            switch (datasetName)
            {
                case SampleDataset.HousingRegression:
                    dataFolder = string.Format("{0}\\SiaNet\\housing", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
                    path.Train = string.Format("{0}\\train.csv", dataFolder);
                    path.Test = string.Format("{0}\\test.csv", dataFolder);
                    break;
                case SampleDataset.MNIST:
                    dataFolder = string.Format("{0}\\SiaNet\\MNIST", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
                    path.Train = string.Format("{0}\\train.txt", dataFolder);
                    path.Test = string.Format("{0}\\test.txt", dataFolder);
                    break;
                case SampleDataset.Cifar10:
                    dataFolder = string.Format("{0}\\SiaNet\\Cifar10", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
                    path.Train = string.Format("{0}\\train.txt", dataFolder);
                    path.Test = string.Format("{0}\\test.txt", dataFolder);
                    break;
                case SampleDataset.Cifar100:
                    dataFolder = string.Format("{0}\\SiaNet\\Cifar100", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
                    path.Train = string.Format("{0}\\train.txt", dataFolder);
                    path.Test = string.Format("{0}\\test.txt", dataFolder);
                    break;
                default:
                    break;
            }

            return path;
        }

        private static void CheckAndDownloadHousingRegression(bool force = false)
        {
            string dataFolder = string.Format("{0}\\SiaNet\\housing", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
            if(force)
            {
                Directory.Delete(dataFolder);
            }

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

        private static void CheckAndDownloadMNIST(bool force = false)
        {
            string dataFolder = string.Format("{0}\\SiaNet\\MNIST", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
            if (force)
            {
                Directory.Delete(dataFolder);
            }

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

        private static void CheckAndDownloadCifar10(bool force = false)
        {
            string dataFolder = string.Format("{0}\\SiaNet\\Cifar10", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
            if (force)
            {
                Directory.Delete(dataFolder);
            }

            if (!Directory.Exists(dataFolder))
            {
                Directory.CreateDirectory(dataFolder);
            }

            string localfile = string.Format("{0}\\train.txt", dataFolder);
            if (!System.IO.File.Exists(localfile))
            {
                DownloadFile(string.Format(cifar10DataUrl, "train.txt"), localfile);
            }

            localfile = string.Format("{0}\\test.txt", dataFolder);
            if (!System.IO.File.Exists(localfile))
            {
                DownloadFile(string.Format(cifar10DataUrl, "test.txt"), localfile);
            }
        }

        private static void DownloadFile(string serverPath, string localPath)
        {
            WebClient wb = new WebClient();
            wb.DownloadFile(serverPath, localPath);
        }
    }
}
