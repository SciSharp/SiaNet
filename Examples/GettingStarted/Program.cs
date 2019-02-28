using System;
using SiaNet;
using SiaNet.Backend.ArrayFire;
using SiaNet.Data;
using SiaNet.Engine;
using SiaNet.Initializers;
using SiaNet.Layers;

namespace GettingStarted
{
    class Program
    {
        static void Main(string[] args)
        {
            //Setup Engine
            Global.UseEngine(ArrayFireBackend.Instance, DeviceType.CPU);

            //Prep Data
            var (x, y) = PrepDataset();
            DataFrameIter trainSet = new DataFrameIter(x, y);
            
            //Build model with simple fully connected layers
            var model = new Sequential();
            model.EpochEnd += Model_EpochEnd;
            model.Add(new Dense(4, ActType.ReLU));
            model.Add(new Dense(2, ActType.ReLU));
            model.Add(new Dense(1, ActType.ReLU));

            //Compile with Optimizer, Loss and Metric
            model.Compile(OptimizerType.Adam, LossType.BinaryCrossEntropy, MetricType.BinaryAccurary);

            // Train for 100 epoch with batch size of 2
            model.Train(trainSet, 100, 4);

            //Create prediction data to evaluate
            DataFrame2D predX = new DataFrame2D(2);
            predX.Load(1, 0, 1, 1); //Result should be 1 and 0

            var rawPred = model.Predict(predX);

            Console.ReadLine();
        }

        private static void Model_EpochEnd(object sender, EpochEndEventArgs e)
        {
            Console.WriteLine("Epoch: {0}, Loss: {1}, Metric: {2}", e.Epoch, e.Loss, e.Metric);
        }

        private static (DataFrame2D, DataFrame2D) PrepDataset()
        {
            // We will prepare XOR gate dataset which will be treated as classification problem.
            // More about this: https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b
            DataFrame2D x = new DataFrame2D(2);
            x.Load(new float[] { 0, 0, 0, 1, 1, 0, 1, 1 });

            DataFrame2D y = new DataFrame2D(1);
            y.Load(new float[] { 0, 1, 1, 0 });

            return (x, y);
        }
    }
}
