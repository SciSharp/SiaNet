using System;
using SiaNet;
using SiaNet.Data;
using SiaNet.Layers;
using TensorSharp;

namespace MNIST
{
    class Program
    {
        static void Main(string[] args)
        {
            //Global.UseGpu();

            string datasetFolder = @"C:\dataset\MNIST";

            var ((trainX, trainY), (valX, valY)) = MNISTParser.LoadDataSet(datasetFolder, 6000, 1000);
            Console.WriteLine("Train and Test data loaded");
            DataFrameIter trainIter = new DataFrameIter(trainX, trainY);
            DataFrameIter valIter = new DataFrameIter(valX, valY);

            Sequential model = BuildConvModel();

            model.Compile(OptimizerType.Adam, LossType.CategorialCrossEntropy, MetricType.Accuracy);
            Console.WriteLine("Model compiled.. initiating training");

            model.EpochEnd += Model_EpochEnd;

            model.Train(trainIter, 10, 200, valIter);

            //var x = Tensor.FromArray(Global.Device, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            //x = x.Reshape(1, 2, 3, 3);
            //var (n, d, h, w) = x.GetConv2DShape();
            //x = x.Reshape(n * d, 1, h, w);
            //x.Print();
            //var X_col = ImgUtil.Im2Col(x, new Tuple<uint, uint>(3, 3), padding: 1, stride: 1);
            ////X_col.Print();
            //var max_idx = TOps.Max(X_col, 0);
            //max_idx.Print();

            //Console.ReadLine();
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
            model.Add(new Conv2D(filters: 30, kernalSize: Tuple.Create<uint, uint>(5, 5), activation: ActType.ReLU));
            model.Add(new MaxPooling2D(poolSize: Tuple.Create<uint, uint>(2, 2)));
            model.Add(new Conv2D(filters: 15, kernalSize: Tuple.Create<uint, uint>(3, 3), activation: ActType.ReLU));
            model.Add(new MaxPooling2D(poolSize: Tuple.Create<uint, uint>(2, 2)));
            model.Add(new Dropout(0.2f));
            model.Add(new Flatten());
            model.Add(new Dense(dim: 128, activation: ActType.ReLU));
            model.Add(new Dense(dim: 50, activation: ActType.ReLU));
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
