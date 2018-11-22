namespace Samples.Common
{
    using SiaNet.Common;
    using SiaNet;
    using SiaNet.Data;
    using SiaNet.Layers;
    using SiaNet.Optimizers;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    internal class MNISTClassifier
    {
        private static DataFrameList<float> train;
        private static DataFrameList<float> test;

        private static Sequential model;
        static int imgDim = 28 * 28;
        static int labelDim = 10;
        public static void LoadData()
        {
            Downloader.DownloadSample(SampleDataset.MNIST);
            var samplePath = Downloader.GetSamplePath(SampleDataset.MNIST);

            train = new DataFrameList<float>(DataFrame<float>.LoadBinary(samplePath.TrainX, 60000, imgDim), DataFrame<float>.LoadBinary(samplePath.TrainY, 60000, labelDim));
            test = new DataFrameList<float>(DataFrame<float>.LoadBinary(samplePath.TestX, 10000, imgDim), DataFrame<float>.LoadBinary(samplePath.TestY, 10000, labelDim));
        }

        public static void BuildModel(bool useConvolution = true)
        {
            if (useConvolution)
            {
                BuildConvolutionLayer();
            }
            else
            {
                BuildMLP();
            }
        }

        private static void BuildMLP()
        {
            model = new Sequential(new Shape(imgDim));
            model.Add(new Dense(dim: 200, activation: new SiaNet.Layers.Activations.ReLU()));
            model.Add(new Dense(dim: 400, activation: new SiaNet.Layers.Activations.ReLU()));
            model.Add(new Dropout(0.2));
            model.Add(new Dense(dim: labelDim));
        }

        private static void BuildConvolutionLayer()
        {
            train.Features.Reshape(new Shape(28, 28,1 ));
            
            model = new Sequential(new Shape(28, 28, 1));
           
            model.Add(new Conv2D(channels: 4, kernalSize: Tuple.Create(3, 3), strides: Tuple.Create(2, 2), activation: new SiaNet.Layers.Activations.ReLU(), weightInitializer: new SiaNet.Initializers.Xavier(), useBias: true, biasInitializer: new SiaNet.Initializers.Ones()));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Conv2D(channels: 8, kernalSize: Tuple.Create(3, 3), strides: Tuple.Create(2, 2), activation: new SiaNet.Layers.Activations.ReLU(), weightInitializer: new SiaNet.Initializers.Xavier()));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Dense(labelDim));
        }

        public static void Train()
        {
            var compiledModel = model.Compile();
            compiledModel.EpochEnd += CompiledModel_EpochEnd;
            compiledModel.Fit(train, 10, 32, optimizer: new SiaNet.Optimizers.SGD(learningRate: 0.01), lossMetric: new SiaNet.Metrics.CrossEntropy(), evaluationMetric: new SiaNet.Metrics.Accuracy(), shuffle: false);
        }

        private static void CompiledModel_EpochEnd(object sender, SiaNet.EventArgs.EpochEndEventArgs e)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, Acc: {2}", e.Epoch, e.Loss, e.Metric));
        }
    }
}
