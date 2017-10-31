using SiaNet.Common;
using SiaNet.Model;
using SiaNet.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Examples
{
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
            model.Add(new Conv2D(Tuple.Create(28, 28, 1), 4, Tuple.Create(3, 3), Tuple.Create(2, 2), activation: OptActivations.None, weightInitializer: OptInitializers.Xavier));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Conv2D(8, Tuple.Create(3, 3), Tuple.Create(2, 2), activation: OptActivations.None, weightInitializer: OptInitializers.Xavier));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Dense(numClasses));
        }

        public static void Train()
        {
            model.Compile(OptOptimizers.SGD, OptLosses.CrossEntropy, OptMetrics.Accuracy);
            model.Train(train, 5, 1000, null);
        }

        private static void Model_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {

        }

        private static void Model_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, Accuracy: {2}", epoch, loss, metrics[OptMetrics.Accuracy]));
        }
    }
}
