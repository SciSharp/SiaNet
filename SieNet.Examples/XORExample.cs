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
    internal class XORExample
    {
        private static XYFrame trainData;

        private static Sequential model;

        public static void LoadData()
        {
            trainData = new XYFrame();

            /*
            //One approach of building dataset
            trainData.Add(new List<float>() { 0, 0 }, 0);
            trainData.Add(new List<float>() { 0, 1 }, 1);
            trainData.Add(new List<float>() { 1, 0 }, 1);
            trainData.Add(new List<float>() { 1, 1 }, 0);
            trainData.YFrame.OneHotEncode();
            */
            
            //Second approach
            trainData.XFrame.Add(0, 0); trainData.YFrame.Add(0);
            trainData.XFrame.Add(0, 1); trainData.YFrame.Add(1);
            trainData.XFrame.Add(1, 0); trainData.YFrame.Add(1);
            trainData.XFrame.Add(1, 1); trainData.YFrame.Add(0);
            trainData.YFrame.OneHotEncode();
        }

        public static void BuildModel()
        {
            model = new Sequential();
            model.Add(new Dense(dim: 2, shape: 2, act: OptActivations.Sigmoid, weightInitializer: new Model.Initializers.Xavier()));
            model.Add(new Dense(dim: 2));

            model.OnEpochEnd += Model_OnEpochEnd;
        }

        public static void Train()
        {
            model.Compile(OptOptimizers.SGD, OptLosses.CrossEntropy, OptMetrics.Accuracy);
            model.Train(trainData, 100, 2);
        }

        private static void Model_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, Acc: {2}", epoch, loss, metrics.First().Value));
        }
    }
}
