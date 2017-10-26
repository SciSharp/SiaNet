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

namespace SiaNet.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            var device = DeviceDescriptor.CPUDevice;
            if (args.Length > 0)
            {
                if(args[0].ToUpper() == "GPU")
                {
                    int gpuNum = 0;
                    if (args.Length == 2)
                    {
                        int.TryParse(args[1], out gpuNum);
                    }

                    device = DeviceDescriptor.GPUDevice(gpuNum);
                }
            }

            //Housing regression example
            HousingRegression.LoadData();
            HousingRegression.BuildModel();
            HousingRegression.Train();

            //Housing regression example
            //MNISTClassifier.LoadData();
            //MNISTClassifier.BuildModel();
            //MNISTClassifier.Train();

            Console.ReadLine();
        }

       
    }

    
}
