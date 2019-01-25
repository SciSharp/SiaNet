using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using TensorSharp;
using SiaNet;
using SiaNet.Data;
using TensorSharp.Expression;
using SiaNet.Layers;
using SiaNet.Initializers;

namespace BasicTest
{
    public class DataSet
    {
        public Tensor inputs;
        public Tensor targets;
        public Tensor targetValues;
    }

    public class MNIST
    {
        private const string MnistTrainImages = "train-images.idx3-ubyte";
        private const string MnistTrainLabels = "train-labels.idx1-ubyte";
        private const string MnistTestImages = "t10k-images.idx3-ubyte";
        private const string MnistTestLabels = "t10k-labels.idx1-ubyte";

        private static (ImageFrame, ImageFrame) trainingData;
        private static (ImageFrame, ImageFrame) testingData;

        public static void Run()
        {
            string datasetFolder = @"C:\dataset\MNIST";
            Console.WriteLine("MNIST Example started...");
            LoadDataSet(datasetFolder);
            Console.WriteLine("Train and Test data loaded");
            DataFrameIter trainIter = new DataFrameIter(trainingData.Item1, trainingData.Item2);
            DataFrameIter valIter = new DataFrameIter(testingData.Item1, testingData.Item2);

            Sequential model = new Sequential();
            model.Add(new Dense(dim: 784, activation: ActivationType.ReLU, kernalInitializer: new GlorotUniform()));
            model.Add(new Dense(dim: 10, activation: ActivationType.Softmax, kernalInitializer: new GlorotUniform()));

            //model.Add(new Dense(dim: 784, kernalInitializer: new GlorotUniform()));
            //model.Add(new SiaNet.Layers.Activations.Relu());
            //model.Add(new Dense(dim: 10, kernalInitializer: new GlorotUniform()));
            //model.Add(new SiaNet.Layers.Activations.Softmax());

            model.Compile(OptimizerType.Adam, LossType.CategorialCrossEntropy, MetricType.Accuracy);
            Console.WriteLine("Model compiled.. initiating training");

            model.EpochEnd += Model_EpochEnd;

            model.Train(trainIter, 25, 32);
        }

        private static void Model_EpochEnd(object sender, EpochEndEventArgs e)
        {
            Console.WriteLine("Epoch: {0}, Samples/sec: {1}, Loss: {2}, Metric: {3}, Elapse: {4}", e.Epoch, e.SamplesSeen, e.Loss, e.Metric, e.Duration);
        }

        static void LoadDataSet(string baseFolder)
        {
            var trainingImages = MnistParser.Parse(
                Path.Combine(baseFolder, MnistTrainImages),
                Path.Combine(baseFolder, MnistTrainLabels),
                600);

            var testImages = MnistParser.Parse(
                Path.Combine(baseFolder, MnistTestImages),
                Path.Combine(baseFolder, MnistTestLabels),
                100);

            trainingData = BuildSet(trainingImages);
            testingData = BuildSet(testImages);
        }

        public static (ImageFrame, ImageFrame) BuildSet(DigitImage[] images)
        {
            var cpuAllocator = new TensorSharp.Cpu.CpuAllocator();

            var inputs = new Tensor(Global.Device, DType.Float32, images.Length, MnistParser.ImageSize, MnistParser.ImageSize);
            var outputs = new Tensor(Global.Device, DType.Float32, images.Length, 10);

            for (int i = 0; i < images.Length; ++i)
            {
                var target = inputs.Select(0, i);

                Variable.FromArray(images[i].pixels, Global.Device)
                    .AsType(DType.Float32)
                    .Evaluate(target);

                target = target / 255;
            }

            Ops.FillOneHot(outputs, MnistParser.LabelCount, images.Select(x => (int)x.label).ToArray());
            inputs = inputs.View(images.Length, 784);
            return (new ImageFrame(inputs), new ImageFrame(outputs));
        }
    }

    public class DigitImage
    {
        public readonly byte[,] pixels;
        public readonly byte label;

        public DigitImage(byte[,] pixels, byte label)
        {
            this.pixels = (byte[,])pixels.Clone();
            this.label = label;
        }
    }

    public static class MnistParser
    {
        public const int ImageSize = 28;
        public const int LabelCount = 10; // each digit has one of 10 possible labels

        public static DigitImage[] Parse(string imageFile, string labelFile, int? maxImages)
        {
            var result = new List<DigitImage>();

            using (var brLabels = new BinaryReader(new FileStream(labelFile, FileMode.Open)))
            using (var brImages = new BinaryReader(new FileStream(imageFile, FileMode.Open)))
            {
                int magic1 = SwapEndian(brImages.ReadInt32());
                int numImages = SwapEndian(brImages.ReadInt32());
                int numRows = SwapEndian(brImages.ReadInt32());
                int numCols = SwapEndian(brImages.ReadInt32());

                int magic2 = SwapEndian(brLabels.ReadInt32());
                int numLabels = SwapEndian(brLabels.ReadInt32());

                var pixels = new byte[ImageSize, ImageSize];

                var images = maxImages.HasValue ? Math.Min(maxImages.Value, numImages) : numImages;

                for (int di = 0; di < images; ++di)
                {
                    for (int i = 0; i < ImageSize; ++i)
                    {
                        for (int j = 0; j < ImageSize; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i, j] = b;
                        }
                    }

                    byte lbl = brLabels.ReadByte();

                    result.Add(new DigitImage(pixels, lbl));
                }
            }

            return result.ToArray();
        }

        private static int SwapEndian(int x)
        {
            return (int)SwapBytes((uint)x);
        }
        private static uint SwapBytes(uint x)
        {
            // swap adjacent 16-bit blocks
            x = (x >> 16) | (x << 16);
            // swap adjacent 8-bit blocks
            return ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
        }
    }
}
