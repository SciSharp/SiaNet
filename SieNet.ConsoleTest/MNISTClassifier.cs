using SiaNet.Model;
using SiaNet.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Test
{
    internal class MNISTClassifier
    {
        private static TrainTestFrame traintest;

        private static Sequential model;

        public static void LoadData()
        {
            DataFrame frame = new DataFrame();
            string trainFile = AppDomain.CurrentDomain.BaseDirectory + "\\samples\\housing\\train.csv";
            frame.LoadFromCsv(trainFile);
            var xy = frame.SplitXY(14, new[] { 1, 13 });
            traintest = xy.SplitTrainTest(0.25);
        }

        public static void BuildModel(bool useConvolution = true)
        {
            model = new Sequential();
            model.OnEpochEnd += Model_OnEpochEnd;
            model.OnTrainingEnd += Model_OnTrainingEnd;

            int[] imageDim = useConvolution ? new int[] { 28, 28, 1 } : new int[] { 784 };
            int numClasses = 10;

            if (useConvolution)
            {
                BuildConvolutionLayer(imageDim, numClasses);
            }
            else
            {
                BuildMLP(imageDim, numClasses);
            }
        }

        private static void BuildMLP(int[] imageDim, int numClasses)
        {
            model.Add(new Dense(200, imageDim[0], OptActivations.Sigmoid));
            model.Add(new Dense(numClasses));
        }

        private static void BuildConvolutionLayer(int[] imageDim, int numClasses)
        {
            model.Add(new Conv2D(4, Tuple.Create(3, 3), Tuple.Create(2, 2), activation: OptActivations.ReLU, weightInitializer: OptInitializers.Xavier));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Conv2D(8, Tuple.Create(3, 3), Tuple.Create(2, 2), activation: OptActivations.ReLU, weightInitializer: OptInitializers.Xavier));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Dense(numClasses));
        }

        public static void Train()
        {
            model.Compile(OptOptimizers.SGD, OptLosses.CrossEntropy, OptMetrics.Accuracy);
            model.Train(traintest.Train, 32, 100, traintest.Test);
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
