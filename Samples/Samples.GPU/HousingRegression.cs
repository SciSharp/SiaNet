using SiaNet.Common;
using SiaNet.Model;
using SiaNet.Model.Data;
using SiaNet.Model.Initializers;
using SiaNet.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Samples.GPU
{
  
    internal class HousingRegression
    {
        private static DataFrameList traintest;

        private static Sequential model;

        public static void LoadData()
        {
            CsvDataFrame frame = new CsvDataFrame();
            Downloader.DownloadSample(SampleDataset.HousingRegression);
            var samplePath = Downloader.GetSamplePath(SampleDataset.HousingRegression);
            frame.ReadCsv(samplePath.Train, true);
            
        }

        public static void BuildModel()
        {
            model = new Sequential(new Shape(10));
            model.Add(new Dense(dim: 20, activation: new Model.Layers.Activations.ReLU()));
            model.Add(new Dense(dim: 20, activation: new Model.Layers.Activations.ReLU()));
            model.Add(new Dense(dim: 1));
        }

        public static void Train()
        {
            var compiledModel = model.Compile();
            compiledModel.TrainingEnd += CompiledModel_TrainingEnd;
            compiledModel.Fit(traintest, 100, 32, optimizer: new Model.Optimizers.Adam(), lossMetric: new Model.Metrics.MeanSquaredError(), evaluationMetric: new Model.Metrics.MeanAbsoluteError(), shuffle: true);
        }

        private static void CompiledModel_TrainingEnd(object sender, EventArgs.TrainingEndEventArgs e)
        {
            Console.WriteLine("Training completed. Mean: {0}, Std: {1}", e.Loss, e.Metric);
        }
    }
}
