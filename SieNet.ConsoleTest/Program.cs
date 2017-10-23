using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet;
using SiaNet.Model;
using SiaNet.Model.Layers;
using CNTK;
using SiaNet.Processing;

namespace SiaNet.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            GlobalParameters.Device = CNTK.DeviceDescriptor.CPUDevice;
            //var c1 = SiaNet.NN.Convolution.Conv2D(Tuple.Create<int, int, int>(28, 28, 1), 4, Tuple.Create<int, int>(3, 3), Tuple.Create<int, int>(2, 2), true, null, OptActivations.ReLU, true);
            //var c2 = SiaNet.NN.Convolution.Conv2D(c1, 3, Tuple.Create<int, int>(2, 2), Tuple.Create<int, int>(1, 1), true);
            //var l1 = SiaNet.NN.Basic.Dense(c2, 5, OptActivations.ReLU, true, OptInitializers.Xavier, OptInitializers.Ones);
            DataFrame frame = new DataFrame();
            frame.LoadFromCsv(@"D:\work\SiaCog\Sia.DNN\ConsolApp.Test\Samples\housing\train.csv", dataType: DataType.Float);

            var xy = frame.SplitXY(14, new[] { 1, 13});
           

            var model = new Sequential();
            
            model.Add(new Dense(13, 12, OptActivations.ReLU));
            model.Add(new Dense(13, OptActivations.Tanh));
            model.Add(new Dropout(0.2f));
            model.Add(new Dense(1));

            model.Compile(OptOptimizers.Adam, OptLosses.MeanSquaredError, OptMetrics.MAE);
            model.OnEpochEnd += Model_OnEpochEnd;
            model.Train(xy, 100, 10);
        }

        private static void Model_OnEpochEnd(int epoch, float loss, Dictionary<string, float> metrics)
        {
            int e = epoch;
        }
    }
}
