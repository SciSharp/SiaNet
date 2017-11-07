using SiaNet.Common;
using SiaNet.Model;
using SiaNet.Model.Layers;
using SiaNet.Model.Optimizers;
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
            model.Add(new Dense(200, imageDim[0], OptActivations.ReLU));
            model.Add(new Dense(400, OptActivations.ReLU));
            model.Add(new Dropout(0.2));
            model.Add(new Dense(numClasses));
        }

        private static void BuildConvolutionLayer(int[] imageDim, int numClasses)
        {
            model.Add(new Conv2D(Tuple.Create(imageDim[0], imageDim[1], imageDim[2]), 4, Tuple.Create(3, 3), Tuple.Create(2, 2), activation: OptActivations.ReLU, weightInitializer: OptInitializers.Xavier, useBias: true, biasInitializer: OptInitializers.Ones));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Conv2D(8, Tuple.Create(3, 3), Tuple.Create(2, 2), activation: OptActivations.ReLU, weightInitializer: OptInitializers.Xavier));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Dense(numClasses));
        }

        public static void Train()
        {
            //model.Compile(OptOptimizers.SGD, OptLosses.CrossEntropy, OptMetrics.Accuracy);
            model.Compile(new SGD(0.1), OptLosses.CrossEntropy, OptMetrics.Accuracy, Regulizers.RegL2(0.1));
            model.Train(train, 10, 64, null);
        }

        private static void Model_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {

        }

        private static void Model_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, Accuracy: {2}", epoch, loss, metrics.First().Value));
        }
    }
}
