using SiaDNN.Initializers;
using SiaNet;
using SiaNet.Backend;
using SiaNet.Data;
using SiaNet.Layers;
using System;

namespace ORGate
{
    class Program
    {
        static void Main(string[] args)
        {
            GlobalParam.Device = Context.Cpu();

            //Pred data
            DataFrame train_x = new DataFrame(2);
            DataFrame train_y = new DataFrame(1);
            train_x.AddData(0, 0);
            train_x.AddData(0, 1);
            train_x.AddData(1, 0);
            train_x.AddData(1, 1);

            train_y.AddData(0);
            train_y.AddData(1);
            train_y.AddData(1);
            train_y.AddData(1);

            DataFrameIter train = new DataFrameIter(train_x.ToVariable(), train_y.ToVariable());

            //Build Model
            Sequential model = new Sequential(2);
            model.Add(new Dense(dim: 4, activation: ActivationType.ReLU, kernalInitializer: new GlorotUniform()));
            model.Add(new Dense(dim: 1));

            //Train
            model.Compile(OptimizerType.Adam, LossType.BinaryCrossEntropy, MetricType.Accuracy);
            model.Fit(train, 100, 2);

            Console.ReadLine();
        }
        
    }
}
