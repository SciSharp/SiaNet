using System;
using SiaNet;
using SiaNet.Data;
using SiaNet.Initializers;
using SiaNet.Layers;
using TensorSharp;
using TensorSharp.Expression;

namespace MNIST
{
    class Program
    {
        static void Main(string[] args)
        {
            Global.UseGpu();

            Tensor x = Tensor.FromArray(Global.Device, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            x = x.Reshape(3, 3);

            var result = TOps.Diag(x);
            result.Print();

            string datasetFolder = @"C:\dataset\MNIST";
            bool useDenseModel = false;

            var ((trainX, trainY), (valX, valY)) = MNISTParser.LoadDataSet(datasetFolder, trainCount: 60000, testCount: 10000, flatten: useDenseModel);
            Console.WriteLine("Train and Test data loaded");
            DataFrameIter trainIter = new DataFrameIter(trainX, trainY);
            DataFrameIter valIter = new DataFrameIter(valX, valY);

            Sequential model = null;
            if (useDenseModel)
                model = BuildFCModel();
            else
                model = BuildConvModel();

            model.Compile(OptimizerType.Adam, LossType.CategorialCrossEntropy, MetricType.Accuracy);
            Console.WriteLine("Model compiled.. initiating training");

            model.EpochEnd += Model_EpochEnd;

            model.Train(trainIter, 10, 32, valIter);

            Console.ReadLine();
        }

        private static Sequential BuildFCModel()
        {
            Sequential model = new Sequential();
            model.Add(new Dense(dim: 784, activation: ActType.ReLU));
            model.Add(new Dense(dim: 10, activation: ActType.Softmax));

            return model;
        }

        private static Sequential BuildConvModel()
        {
            Sequential model = new Sequential();
            model.Add(new Conv2D(filters: 16, kernalSize: Tuple.Create<uint, uint>(5, 5), activation: ActType.ReLU));
            model.Add(new MaxPooling2D(poolSize: Tuple.Create<uint, uint>(2, 2)));
            model.Add(new Conv2D(filters: 32, kernalSize: Tuple.Create<uint, uint>(5, 5), activation: ActType.ReLU));
            model.Add(new MaxPooling2D(poolSize: Tuple.Create<uint, uint>(2, 2)));
            //model.Add(new Dropout(0.2f));
            model.Add(new Flatten());
            model.Add(new Dense(dim: 128, activation: ActType.Sigmoid));
            model.Add(new Dense(dim: 10, activation: ActType.Softmax));

            return model;
        }

        private static void Model_EpochEnd(object sender, EpochEndEventArgs e)
        {
            if(e.ValidationLoss > 0)
                Console.WriteLine("Epoch: {0}, Samples/sec: {1}, Loss: {2}, Acc: {3}, Val_Loss: {4}, Val_Acc: {5}, Elapse: {6}", e.Epoch, e.SamplesSeen, e.Loss, e.Metric, e.ValidationLoss, e.ValidationMetric, e.Duration);
            else
                Console.WriteLine("Epoch: {0}, Samples/sec: {1}, Loss: {2}, Acc: {3}, Elapse: {4}", e.Epoch, e.SamplesSeen, e.Loss, e.Metric, e.Duration);
        }
    }
}
