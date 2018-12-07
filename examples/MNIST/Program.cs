using SiaDNN.Initializers;
using SiaNet;
using SiaNet.Backend;
using SiaNet.Data;
using SiaNet.Layers;
using System;

namespace MNIST
{
    class Program
    {
        static void Main(string[] args)
        {
            GlobalParam.Device = Context.Cpu();

            int inputDim = 28 * 28;
            int labelCount = 10;
            uint batchSize = 32;

            string trainImagePath = "./mnist_data/train-images-idx3-ubyte";
            string trainLabelPath = "./mnist_data/train-labels-idx1-ubyte";
            string valImagePath = "./mnist_data/t10k-images-idx3-ubyte";
            string valLabelPath = "./mnist_data/t10k-labels-idx1-ubyte";

            var (train, val) = DataSetParser.MNIST(trainImagePath, trainLabelPath, valImagePath, valLabelPath);

            var model = new Sequential((uint)inputDim);
            model.Add(new Dense(inputDim, ActivationType.ReLU, new GlorotUniform()));
            model.Add(new Dense(128, ActivationType.ReLU, new GlorotUniform()));
            //model.Add(new Dense(64, ActivationType.ReLU, new GlorotUniform()));
            model.Add(new Dense(labelCount));

            model.Compile(Optimizers.SGD(0.01f), LossType.CategorialCrossEntropy, MetricType.Accuracy);
            model.Fit(train, 10, batchSize, val);
        }
    }
}
