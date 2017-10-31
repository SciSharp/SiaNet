using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet;
using SiaNet.Model;
using SiaNet.Model.Layers;
using CNTK;
using SiaNet.Processing;
using SiaNet.Common;

namespace SiaNet.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                var devices = DeviceDescriptor.AllDevices().Where(x=>(x.Type == DeviceKind.GPU)).ToList();
                if (devices.Count == 0)
                    throw new Exception("No GPU Device found. Please run the CPU examples instead!");

                Logging.OnWriteLog += Logging_OnWriteLog;

                //Setting global device
                GlobalParameters.Device = devices[0];

                /*
                //Housing regression example
                HousingRegression.LoadData();
                HousingRegression.BuildModel();
                HousingRegression.Train();

                //MNIST Classification example
                MNISTClassifier.LoadData();
                MNISTClassifier.BuildModel();
                MNISTClassifier.Train();
                
                //Image classification example
                Console.WriteLine("ResNet50 Prediction: " + ImageClassification.ImagenetTest(Common.ImageNetModel.ResNet50)[0].Name);
                Console.WriteLine("Cifar 10 Prediction: " + ImageClassification.Cifar10Test(Common.Cifar10Model.ResNet110)[0].Name);
                */

                //Object Detection
                //ObjectDetection.PascalDetection();
                ObjectDetection.GroceryDetection();
                Console.ReadLine();
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.ToString());
                Console.ReadLine();
            }
        }

        private static void Logging_OnWriteLog(string message)
        {
            Console.WriteLine("Log Message: " + message);
        }
    }
}
