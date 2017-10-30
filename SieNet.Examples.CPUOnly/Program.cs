using CNTK;
using SiaNet;
using SiaNet.Examples;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SieNet.Examples.CPUOnly
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                //Setting global device
                GlobalParameters.Device = DeviceDescriptor.CPUDevice;
                Downloader.DownloadSample(SampleDataset.Cifar10);
                //Housing regression example
                HousingRegression.LoadData();
                HousingRegression.BuildModel();
                HousingRegression.Train();

                //MNIST Classification example
                MNISTClassifier.LoadData();
                MNISTClassifier.BuildModel();
                MNISTClassifier.Train();

                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                Console.ReadLine();
            }
        }
    }
}
