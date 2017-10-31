using CNTK;
using SiaNet;
using SiaNet.Application;
using SiaNet.Common;
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
                Logging.OnWriteLog += Logging_OnWriteLog;

                ImageNet app = new ImageNet(ImageNetModel.ResNet50);
                app.LoadModel();
                app.Predict(@"\\userAB\B\batt0153\prefs\Desktop\dog_cls-300x160.jpg");

                /*
                //Housing regression example
                HousingRegression.LoadData();
                HousingRegression.BuildModel();
                HousingRegression.Train();

                //MNIST Classification example
                MNISTClassifier.LoadData();
                MNISTClassifier.BuildModel();
                MNISTClassifier.Train();
                */
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                Console.ReadLine();
            }
        }

        private void EvalImageNet()
        {

        }

        private static void Logging_OnWriteLog(string message)
        {
            Console.WriteLine(message);
        }
    }
}
