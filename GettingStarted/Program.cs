using System;
using SiaNet;
using SiaNet.Data;
using SiaNet.Layers;

namespace GettingStarted
{
    class Program
    {
        static void Main(string[] args)
        {
            Global.UseGpu();
            var (x, y) = PrepDataset();

            DataFrameIter trainSet = new DataFrameIter(x, y);

            //Build model with simple fully connected layers
            var model = new Sequential();
            model.EpochEnd += Model_EpochEnd;
            model.Add(new Dense(2, ActivationType.ReLU));
            model.Add(new Dense(1));

            model.Compile(OptimizerType.Adam, LossType.BinaryCrossEntropy, MetricType.Accuracy);
            model.Train(trainSet, 100, 4);

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
            x.Load(0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f);
            x.Print();

            DataFrame2D y = new DataFrame2D(1);
            y.Load(0f, 1f, 0f, 1f);
            y.Print();

            return (x, y);
        }
    }
}
