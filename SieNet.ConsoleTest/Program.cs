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
            GlobalParameters.Device = CNTK.DeviceDescriptor.CPUDevice;
            DataFrame frame = new DataFrame();
            string trainFile = AppDomain.CurrentDomain.BaseDirectory + "\\samples\\housing\\train.csv";
            frame.LoadFromCsv(trainFile);

            var xy = frame.SplitXY(14, new[] { 1, 13});
            var traintest = xy.SplitTrainTest(0.25);

            var model = new Sequential();
            model.OnEpochEnd += Model_OnEpochEnd;
            model.OnTrainingEnd += Model_OnTrainingEnd;
            model.Add(new Dense(13, 12, OptActivations.ReLU));
            model.Add(new Dense(13, OptActivations.ReLU));
            model.Add(new Dense(1));
            model.Compile(OptOptimizers.Adam, OptLosses.MeanSquaredError, OptMetrics.MAE);
            
            model.Train(traintest.Train, 64, 200, traintest.Test);
            Console.ReadLine();
        }

        private static void Model_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {
            var mean = trainingResult[OptMetrics.MAE].Mean();
            var std = trainingResult[OptMetrics.MAE].Std();
            Console.WriteLine("Training completed. Mean: {0}, Std: {1}", mean, std);
        }

        private static void Model_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, Accuracy: {2}", epoch, loss, metrics["val_mae"]));
        }
    }

    
}
