using SiaNet.Common;
using SiaNet.Model;
using SiaNet.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Examples
{
    internal class TimeSeriesPrediction
    {
        private static XYFrame train;

        private static Sequential model;

        private static string filepath = string.Format("{0}\\samples\\int_airline_pass.csv", AppDomain.CurrentDomain.BaseDirectory);

        private static int lookback = 1;

        public static void LoadData()
        {
            DataFrame frame = new DataFrame();
            frame.LoadFromCsv(filepath);
            frame = frame.Partition(1);
            frame.Normalize();
            train = frame.ConvertTimeSeries(lookback);
        }

        public static void BuildModel()
        {
            model = new Sequential();
            model.Add(new LSTM(dim: 4, shape: new int[] { lookback }, cellDim: 2));
            model.Add(new Dense(dim: 1));

            model.OnEpochEnd += Model_OnEpochEnd;
            model.OnTrainingEnd += Model_OnTrainingEnd;
        }

        public static void Train()
        {
            model.Compile(OptOptimizers.Adam, OptLosses.MeanSquaredError, OptMetrics.MSLE);
            model.Train(train, 100, 6);
        }

        private static void Model_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {
            var mean = trainingResult[OptMetrics.MSLE].Mean();
            var std = trainingResult[OptMetrics.MSLE].Std();
            Console.WriteLine("Training completed. Mean: {0}, Std: {1}", mean, std);
        }

        private static void Model_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, MSLE: {2}", epoch, loss, metrics.First().Value));
        }
    }
}
