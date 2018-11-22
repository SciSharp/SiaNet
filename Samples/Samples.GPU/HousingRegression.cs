using SiaNet.Common;
using SiaNet;
using SiaNet.Data;
using SiaNet.Initializers;
using SiaNet.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Samples.Common
{
  
    internal class HousingRegression
    {
        private static DataFrameList<float> trainData;

        private static Sequential model;

        public static void LoadData()
        {
            CsvDataFrame<float> frame = new CsvDataFrame<float>();
            Downloader.DownloadSample(SampleDataset.HousingRegression);
            var samplePath = Downloader.GetSamplePath(SampleDataset.HousingRegression);
            frame.ReadCsv(samplePath.Train, true);
            DataFrame<float> trainX = frame[1, 13];
            DataFrame<float> trainY = frame[14, 1];
            trainData = new DataFrameList<float>(trainX, trainY);
        }

        public static void BuildModel()
        {
            model = new Sequential(new Shape(trainData.Features.DataShape[0]));
            model.Add(new Dense(dim: 20, activation: new SiaNet.Layers.Activations.ReLU()));
            model.Add(new Dense(dim: 20, activation: new SiaNet.Layers.Activations.ReLU()));
            model.Add(new Dense(dim: trainData.Labels.DataShape[0]));
        }

        public static void Train()
        {
            var compiledModel = model.Compile();
            compiledModel.TrainingEnd += CompiledModel_TrainingEnd;
            compiledModel.EpochEnd += CompiledModel_EpochEnd;
            compiledModel.Fit(trainData, 500, 32, optimizer: new SiaNet.Optimizers.Adam(), lossMetric: new SiaNet.Metrics.MeanSquaredError(), evaluationMetric: new SiaNet.Metrics.MeanAbsoluteError(), shuffle: true);
        }

        private static void CompiledModel_EpochEnd(object sender, SiaNet.EventArgs.EpochEndEventArgs e)
        {
            Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}", e.Epoch, e.Loss));
        }

        private static void CompiledModel_TrainingEnd(object sender, SiaNet.EventArgs.TrainingEndEventArgs e)
        {
            Console.WriteLine("Training completed. Mean: {0}, Std: {1}", e.Loss, e.Metric);
        }
    }
}
