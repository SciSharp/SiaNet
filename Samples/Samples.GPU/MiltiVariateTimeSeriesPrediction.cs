using SiaNet.Common;
using SiaNet;
using SiaNet.Data;
using SiaNet.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Samples.GPU
{
    internal class MiltiVariateTimeSeriesPrediction
    {
        private static XYFrame train;

        private static Sequential model;

        private static string filepath = string.Format("{0}\\samples\\PRSA_data_multivariate.csv", AppDomain.CurrentDomain.BaseDirectory);

        private static int lookback = 5;

        public static void LoadData()
        {
            DataFrame frame = new DataFrame();
            frame.LoadFromCsv(filepath);
            frame = frame.GetFrame(5);
            frame.Normalize();
            train = frame.ConvertTimeSeries(lookback);
        }

        public static void BuildModel()
        {
            model = new Sequential();
            model.Add(new Reshape(targetshape: Shape.Create(1, train.XFrame.Shape[1]), shape: Shape.Create(lookback)));
            model.Add(new LSTM(dim: 5, shape: Shape.Create(1, train.XFrame.Shape[1])));
            model.Add(new Dense(dim: 1));

            model.OnEpochEnd += Model_OnEpochEnd;
            model.OnTrainingEnd += Model_OnTrainingEnd;
        }

        public static void Train()
        {
            model.Compile(OptOptimizers.Adam, OptLosses.MeanSquaredError, OptMetrics.MSE);
            model.Train(train, 10, 64);
        }

        private static void Model_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {
            var mean = trainingResult[OptMetrics.MSE].Mean();
            var std = trainingResult[OptMetrics.MSE].Std();
            Console.WriteLine("Training completed. Mean: {0}, Std: {1}", mean * 100, std * 100);
        }

        private static void Model_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, MSLE: {2}", epoch, loss, metrics.First().Value));
        }
    }
}
