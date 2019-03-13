using SiaNet;
using SiaNet.Data;
using SiaNet.Engine;
using SiaNet.Events;
using SiaNet.Layers;
using System;

namespace TrainingWithRandomData
{
    class Program
    {
        static void Main(string[] args)
        {
            //Setup Engine
            Global.UseEngine(SiaNet.Backend.ArrayFire.SiaNetBackend.Instance, DeviceType.CPU);

            //Prep Data
            var (x, y) = PrepDataset();
            DataFrameIter trainSet = new DataFrameIter(x, y);

            //Build model with simple fully connected layers
            var model = new Sequential();
            model.EpochEnd += Model_EpochEnd;
            model.Add(new Dense(100, ActType.ReLU));
            model.Add(new Dense(50, ActType.ReLU));
            model.Add(new Dense(1, ActType.Sigmoid));

            //Compile with Optimizer, Loss and Metric
            model.Compile(OptimizerType.Adam, LossType.MeanSquaredError, MetricType.MAE);

            // Train for 100 epoch with batch size of 2
            model.Train(trainSet, 25, 2);
        }

        private static void Model_EpochEnd(object sender, EpochEndEventArgs e)
        {
            Console.WriteLine("Epoch: {0}, Loss: {1}, Metric: {2}", e.Epoch, e.Loss, e.Metric);
        }

        private static (DataFrame2D, DataFrame2D) PrepDataset()
        {
            var K = Global.CurrentBackend;
            Tensor x = K.RandomNormal(new long[] { 900, 10 }, 0.25f, 1);
            Tensor y = K.RandomUniform(new long[] { 900, 1 }, 0, 25);

            DataFrame2D X = new DataFrame2D(10);
            DataFrame2D Y = new DataFrame2D(1);
            X.Load(x.DataFloat);
            Y.Load(y.DataFloat);
            return (X, Y);
        }
    }
}
