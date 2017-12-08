namespace SiaNet.Examples
{
    using SiaNet.Common;
    using SiaNet.Model;
    using SiaNet.Model.Layers;
    using SiaNet.Model.Optimizers;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    internal class Cifar10Classification
    {
        private static ImageDataGenerator train;

        private static ImageDataGenerator validation;

        private static Sequential model;

        public static void LoadData()
        {
            Downloader.DownloadSample(SampleDataset.Cifar10);
            var samplePath = Downloader.GetSamplePath(SampleDataset.Cifar10);

            train = ImageDataGenerator.FlowFromText(samplePath.Train);
            validation = ImageDataGenerator.FlowFromText(samplePath.Test);
        }

        public static void BuildModel(bool useConvolution = true)
        {
            model = new Sequential();
            model.OnEpochEnd += Model_OnEpochEnd;
            model.OnTrainingEnd += Model_OnTrainingEnd;
            model.OnBatchEnd += Model_OnBatchEnd;
            
            int[] imageDim = useConvolution ? new int[] { 32, 32, 3 } : new int[] { 3072 };
            int numClasses = 10;

            if (useConvolution)
            {
                BuildSmallConvolutionLayer(imageDim, numClasses);
            }
            else
            {
                BuildMLP(imageDim, numClasses);
            }
        }

        private static void BuildMLP(int[] imageDim, int numClasses)
        {
            model.Add(new Dense(dim: 3072, shape: imageDim[0], act: OptActivations.ReLU));
            model.Add(new Dense(dim: 2000, act: OptActivations.ReLU));
            model.Add(new Dropout(0.2));
            model.Add(new Dense(dim: numClasses));
        }

        private static void BuildSmallConvolutionLayer(int[] imageDim, int numClasses)
        {
            model.Add(new Conv2D(shape: Tuple.Create(imageDim[0], imageDim[1], imageDim[2]), channels: 32, kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, weightInitializer: OptInitializers.Xavier, padding: true));
            model.Add(new MaxPool2D(poolSize: Tuple.Create(2, 2)));
            model.Add(new Conv2D(channels: 32, kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, weightInitializer: OptInitializers.Xavier, padding: true));
            model.Add(new MaxPool2D(poolSize: Tuple.Create(2, 2)));
            model.Add(new Dropout(0.2));
            model.Add(new Dense(dim: numClasses, act: OptActivations.Softmax));
        }

        public static void Train()
        {
            //model.Compile(OptOptimizers.SGD, OptLosses.CrossEntropy, OptMetrics.Accuracy);
            model.Compile(new SGD(0.01), OptLosses.CrossEntropy, OptMetrics.Accuracy, Regulizers.RegL2(0.01));
            model.Train(train, 25, 64, null);
        }

        private static void Model_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {

        }

        private static void Model_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            //Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, Accuracy: {2}", epoch, loss, metrics.First().Value));
        }

        private static void Model_OnBatchEnd(int epoch, int batchNumber, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            if (batchNumber % 20 == 0)
                Console.WriteLine(string.Format("Epoch: {0}, Batch: {1}, Loss: {2}, Accuracy: {3}", epoch, batchNumber, loss, metrics.First().Value));
        }
    }
}
