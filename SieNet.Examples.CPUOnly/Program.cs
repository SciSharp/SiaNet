using CNTK;
using SiaNet;
using SiaNet.Application;
using SiaNet.Common;
using SiaNet.Model;
using SiaNet.Examples;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet.Model.Layers;

namespace SieNet.Examples.CPUOnly
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                //Setting global device
                Logging.OnWriteLog += Logging_OnWriteLog;

                
                //XOR Example
                XORExample.LoadData();
                XORExample.BuildModel();
                XORExample.Train();

                //Housing regression example
                HousingRegression.LoadData();
                HousingRegression.BuildModel();
                HousingRegression.Train();
                
                //MNIST Classification example
                MNISTClassifier.LoadData();
                MNISTClassifier.BuildModel();
                MNISTClassifier.Train();
                
                //LSTM Time series example
                TimeSeriesPrediction.LoadData();
                TimeSeriesPrediction.BuildModel();
                TimeSeriesPrediction.Train();
                
                //Multi variate time series prediction
                MiltiVariateTimeSeriesPrediction.LoadData();
                MiltiVariateTimeSeriesPrediction.BuildModel();
                MiltiVariateTimeSeriesPrediction.Train();

                //Cifar - 10 Classification example
                //Cifar10Classification.LoadData();
                //Cifar10Classification.BuildModel();
                //Cifar10Classification.Train();

                //Image classification example
                Console.WriteLine("ResNet50 Prediction: " + ImageClassification.ImagenetTest(SiaNet.Common.ImageNetModel.ResNet50)[0].Name);
                //Console.WriteLine("Cifar 10 Prediction: " + ImageClassification.Cifar10Test(SiaNet.Common.Cifar10Model.ResNet110)[0].Name);
                
                //Object Detection
                ObjectDetection.PascalDetection();
                //ObjectDetection.GroceryDetection();
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

        private static void RunTest()
        {
            Random Rnd = new Random();
            DataFrame trnX_fin = new DataFrame();
            DataFrame trnY_fin = new DataFrame();
            for (int cc = 0; (cc < 100); cc++)
            {
                float[] sngLst = new float[100];
                for (int indx = 0; (indx < 100); indx++)
                {
                    sngLst[indx] = (float)Rnd.NextDouble();
                }

                trnX_fin.Add(sngLst);
            }

            for (int cc = 0; (cc < 100); cc++)
            {
                float[] sngLst = new float[3];
                //  fake one hot just for check
                sngLst[0] = 0;
                sngLst[1] = 1;
                sngLst[2] = 0;
                trnY_fin.Add(sngLst);
            }

            XYFrame XYfrm = new XYFrame();
            XYfrm.XFrame = trnX_fin;
            XYfrm.YFrame = trnY_fin;
            //  Split
            TrainTestFrame trainTestFrame = XYfrm.SplitTrainTest(0.3);
            //  init some values
            int shape_of_input = XYfrm.XFrame.Shape[1];
            int embval = 100;
            int seed = 2;
            Sequential model = new Sequential();
            model.Add(new Reshape(Shape.Create(1, embval), Shape.Create(shape_of_input)));
            model.Add(new LSTM(64, returnSequence: false, cellDim:4, weightInitializer: new SiaNet.Model.Initializers.GlorotUniform(0.05, seed),recurrentInitializer: new SiaNet.Model.Initializers.GlorotUniform(0.05, seed), biasInitializer: new SiaNet.Model.Initializers.GlorotUniform(0.05, seed)));
            model.Add(new Dense(3, act: "sigmoid", useBias: true,  weightInitializer: new SiaNet.Model.Initializers.GlorotUniform(0.05, seed)));
            model.Compile(OptOptimizers.Adam, OptLosses.MeanSquaredError, OptMetrics.Accuracy);
            model.Train(trainTestFrame.Train, 200, 8, trainTestFrame.Test);
        }
    }
}
