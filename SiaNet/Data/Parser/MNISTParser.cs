using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;
using System.Linq;

namespace SiaNet.Data
{
    public class MNISTParser
    {
        internal class DigitImage
        {
            public readonly byte[,] pixels;
            public readonly byte label;

            public DigitImage(byte[,] pixels, byte label)
            {
                this.pixels = (byte[,])pixels.Clone();
                this.label = label;
            }
        }

        private const int ImageSize = 28;
        private const int LabelCount = 10;
        private const string MnistTrainImages = "train-images.idx3-ubyte";
        private const string MnistTrainLabels = "train-labels.idx1-ubyte";
        private const string MnistTestImages = "t10k-images.idx3-ubyte";
        private const string MnistTestLabels = "t10k-labels.idx1-ubyte";

        public static ((ImageFrame, ImageFrame), (ImageFrame, ImageFrame)) LoadDataSet(string baseFolder, int trainCount = 60000, int testCount = 10000, bool flatten = false)
        {
            var trainingImages = ParseFile(
                Path.Combine(baseFolder, MnistTrainImages),
                Path.Combine(baseFolder, MnistTrainLabels),
                trainCount);

            var testImages = ParseFile(
                Path.Combine(baseFolder, MnistTestImages),
                Path.Combine(baseFolder, MnistTestLabels),
                testCount);

            return (BuildSet(trainingImages, flatten), BuildSet(testImages, flatten));
        }

        private static (ImageFrame, ImageFrame) BuildSet(DigitImage[] images, bool flatten = false)
        {
            var cpuAllocator = new TensorSharp.Cpu.CpuAllocator();
            
            var inputs = new Tensor(Global.Device, DType.Float32, images.Length, 1, ImageSize, ImageSize);
            var outputs = new Tensor(Global.Device, DType.Float32, images.Length, 10);

            for (int i = 0; i < images.Length; ++i)
            {
                var target = inputs.Select(0, i);

                Variable.FromArray(images[i].pixels, Global.Device)
                    .AsType(DType.Float32)
                    .Evaluate(target);

                target = target / 255;
            }

            Ops.FillOneHot(outputs, LabelCount, images.Select(x => (int)x.label).ToArray());
            if(flatten)
                inputs = inputs.View(images.Length, 784);
            return (new ImageFrame(inputs), new ImageFrame(outputs));
        }

        private static DigitImage[] ParseFile(string imageFile, string labelFile, int? maxImages)
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
