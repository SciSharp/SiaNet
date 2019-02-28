using SiaNet;
using SiaNet.Data;
using SiaNet.Engine;
using SiaNet.Initializers;
using SiaNet.Layers;
using System;

namespace BostonHousingRegressionExample
{
    class Program
    {
        static void Main(string[] args)
        {
            //Setup Engine
            Global.UseEngine(SiaNet.Backend.ArrayFire.SiaNetBackend.Instance, DeviceType.CPU);

            //Load Train and Test CSV data
            var ds = LoadTrain("./train.csv");
            var test = LoadTest("./test.csv");

            var (train, val) = ds.Split(0.25);
            //Build Model
            var model = new Sequential();
            model.EpochEnd += Model_EpochEnd;
            model.Add(new Dense(64, ActType.ReLU));
            model.Add(new Dense(32, ActType.ReLU));
            model.Add(new Dense(1, ActType.Sigmoid));

            //Compile with Optimizer, Loss and Metric
            model.Compile(OptimizerType.Adam, LossType.MeanSquaredError, MetricType.MAE);

            // Train for 100 epoch with batch size of 32
            model.Train(train, 100, 32, val);

            Console.ReadLine();
        }

        private static void Model_EpochEnd(object sender, EpochEndEventArgs e)
        {
            Console.WriteLine("Epoch: {0}, Loss: {1}, Metric: {2}, Val_Loss: {3}, Val_Metric: {4}", e.Epoch, e.Loss, e.Metric, e.ValidationLoss, e.ValidationMetric);
        }


        public static DataFrameIter LoadTrain(string filename)
        {
            DataFrame2D data = DataFrame2D.ReadCsv(filename, true);
            var x = data[1, 13];
            var y = data[14];

            return new DataFrameIter(x, y);
        }

        public static DataFrameIter LoadTest(string filename)
        {
            DataFrame2D data = DataFrame2D.ReadCsv(filename, true);
            var x = data[1, 13];
            return new DataFrameIter(x);
        }
    }
}
