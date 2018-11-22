namespace Samples.GPU
{
    using SiaNet.Common;
    using SiaNet.Model;
    using SiaNet.Model.Layers;
    using SiaNet.Model.Optimizers;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    internal class MNISTClassifier
    {
        private static ImageDataGenerator train;

        private static ImageDataGenerator validation;

        private static Sequential model;

        public static void LoadData()
        {
            Downloader.DownloadSample(SampleDataset.MNIST);
            var samplePath = Downloader.GetSamplePath(SampleDataset.MNIST);

            train = ImageDataGenerator.FlowFromText(samplePath.Train);
            validation = ImageDataGenerator.FlowFromText(samplePath.Test);
        }

        public static void BuildModel(bool useConvolution = true)
        {
            model = new Sequential();
            model.OnEpochEnd += Model_OnEpochEnd;
            model.OnTrainingEnd += Model_OnTrainingEnd;
            model.OnBatchEnd += Model_OnBatchEnd;

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
            model.Add(new Dense(200, imageDim[0], OptActivations.ReLU));
            model.Add(new Dense(400, act: OptActivations.ReLU));
            model.Add(new Dropout(0.2));
            model.Add(new Dense(numClasses));
        }

        private static void BuildConvolutionLayer(int[] imageDim, int numClasses)
        {
            model.Add(new Conv2D(shape: Tuple.Create(imageDim[0], imageDim[1], imageDim[2]), channels: 4, kernalSize: Tuple.Create(3, 3), strides: Tuple.Create(2, 2), activation: OptActivations.None, weightInitializer: OptInitializers.Xavier, useBias: true, biasInitializer: OptInitializers.Ones));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Conv2D(channels: 8, kernalSize: Tuple.Create(3, 3), strides: Tuple.Create(2, 2), activation: OptActivations.None, weightInitializer: OptInitializers.Xavier));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Dense(numClasses));
        }

        public static void Train()
        {
            //model.Compile(OptOptimizers.SGD, OptLosses.CrossEntropy, OptMetrics.Accuracy);
            model.Compile(new SGD(0.003125), OptLosses.CrossEntropy, OptMetrics.Accuracy);
            model.Train(train, 5, 32, null);
        }

        private static void Model_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {

        }

        private static void Model_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, Accuracy: {2}", epoch, loss, metrics.First().Value));
        }

        private static void Model_OnBatchEnd(int epoch, int batchNumber, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            //if (batchNumber % 100 == 0)
                //Console.WriteLine(string.Format("Epoch: {0}, Batch: {1}, Loss: {2}, Accuracy: {3}", epoch, batchNumber, loss, metrics.First().Value));
        }
    }
}
