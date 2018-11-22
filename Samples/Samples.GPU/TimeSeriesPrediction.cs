using SiaNet.Common;
using SiaNet.Model;
using SiaNet.Model.Data;
using SiaNet.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Samples.GPU
{
    internal class TimeSeriesPrediction
    {
        private static DataFrameList train;

        private static Sequential model;

        private static string filepath = string.Format("{0}\\samples\\int_airline_pass.csv", AppDomain.CurrentDomain.BaseDirectory);

        private static int lookback = 3;

        public static void LoadData()
        {
            var frame = Deedle.Frame.ReadCsv(filepath, hasHeaders: true);
            var X = frame["International airline passengers"];

            //train = frame.ConvertTimeSeries(lookback);
        }

        public static void BuildModel()
        {
            model = new Sequential(new Shape(lookback));
            model.Add(new LSTM(dim: 4, returnSequence: true));
            model.Add(new LSTM(dim: 4));
            model.Add(new Dense(dim: 1));
        }

        public static void Train()
        {
            var compiledModel = model.Compile();
            compiledModel.EpochEnd += CompiledModel_EpochEnd;
            compiledModel.TrainingEnd += CompiledModel_TrainingEnd;
            compiledModel.Fit(train, 100, 5, new Model.Optimizers.Adam(), new Model.Metrics.MeanSquaredError());
        }

        private static void CompiledModel_TrainingEnd(object sender, EventArgs.TrainingEndEventArgs e)
        {
            //Console.WriteLine("Training completed. Mean: {0}, Std: {1}", mean * 100, std * 100);
        }

        private static void CompiledModel_EpochEnd(object sender, EventArgs.EpochEndEventArgs e)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, MSLE: {2}", e.Epoch, e.Loss, e.Metric));
        }
    }
}
