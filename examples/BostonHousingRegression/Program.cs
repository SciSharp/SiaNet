using SiaDNN.Initializers;
using SiaNet;
using SiaNet.Backend;
using SiaNet.Data;
using SiaNet.Layers;
using System;

namespace BostonHousingRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            //Environment.SetEnvironmentVariable("MXNET_ENGINE_TYPE", "NaiveEngine");
            GlobalParam.Device = Context.Cpu();

            //Read Data
            CsvDataFrame trainReader = new CsvDataFrame("./data/train.csv", true);
            trainReader.ReadCsv();
            var trainX = trainReader[1, 14];
            var trainY = trainReader[14, 15];

            CsvDataFrame valReader = new CsvDataFrame("./data/test.csv", true);
            valReader.ReadCsv();

            var valX = valReader[1, 14];
            var valY = valReader[14, 15];

            DataFrameIter train = new DataFrameIter(trainX, trainY);
            DataFrameIter val = new DataFrameIter(valX, valY);

            //Build Model
            var model = new Sequential(13);
            model.Add(new Dense(13, ActivationType.ReLU));
            model.Add(new Dense(20, ActivationType.ReLU));
            model.Add(new Dropout(0.25f));
            model.Add(new Dense(20, ActivationType.ReLU));
            model.Add(new Dense(1));

            model.Compile(OptimizerType.RMSprop, LossType.MeanSquaredError, MetricType.MeanSquaredError);
            model.Fit(train, 200, 16);

            Console.ReadLine();
        }
    }
}
