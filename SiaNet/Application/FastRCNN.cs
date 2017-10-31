using CNTK;
using SiaNet.Common;
using SiaNet.Common.Resources;
using SiaNet.Model;
using SiaNet.Processing;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Dynamic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Application
{
    public class FastRCNN
    {
        FastRCNNModel model;

        Function modelFunc;

        Dictionary<int, string> actualValues;

        public FastRCNN(FastRCNNModel model)
        {
            this.model = model;
            actualValues = new Dictionary<int, string>();
            ExpandoObject jsonValues = Newtonsoft.Json.JsonConvert.DeserializeObject<ExpandoObject>(PredMap.Cifar10);
            foreach (var item in jsonValues)
            {
                actualValues.Add(Convert.ToInt32(item.Key), item.Value.ToString());
            }
        }

        public void LoadModel()
        {
            try
            {
                string modelFile = "";
                string baseFolder = string.Format("{0}\\SiaNet\\models", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
                switch (model)
                {
                    case FastRCNNModel.Pascal:
                        Downloader.DownloadModel(PreTrainedModelPath.FastRCNNPath.Pascal);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.FastRCNNPath.Pascal);
                        break;
                    case FastRCNNModel.Grocery100:
                        Downloader.DownloadModel(PreTrainedModelPath.FastRCNNPath.Grocery100);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.FastRCNNPath.Grocery100);
                        break;
                    default:
                        throw new Exception("Invalid model selected!");
                }

                modelFunc = Function.Load(modelFile, GlobalParameters.Device);
                Logging.WriteTrace("Model loaded.");
            }
            catch (Exception ex)
            {
                Logging.WriteTrace(ex);
                throw ex;
            }
        }

        public List<PredResult> Predict(string imagePath, int topK = 3)
        {
            try
            {
                Bitmap bmp = new Bitmap(Bitmap.FromFile(imagePath));
                var resized = bmp.Resize(1000, 1000, true);
                List<float> resizedCHW = resized.ParallelExtractCHW();
                List<float> roiList = new List<float>();
                string roiCoordinates = "0.219 0.0 0.165 0.29 0.329 0.025 0.07 0.115 0.364 0.0 0.21 0.13 0.484 0.0 0.075 0.06 0.354 0.045 0.055 0.09 0.359 0.075 0.095 0.07 0.434 0.155 0.04 0.085 0.459 0.165 0.145 0.08 0.404 0.12 0.055 0.06 0.714 0.235 0.06 0.12 0.659 0.31 0.065 0.075 0.299 0.16 0.1 0.07 0.449 0.18 0.19 0.15 0.284 0.21 0.135 0.115 0.254 0.205 0.07 0.055 0.234 0.225 0.075 0.095 0.239 0.23 0.07 0.085 0.529 0.235 0.075 0.13 0.229 0.24 0.09 0.085 0.604 0.285 0.12 0.105 0.514 0.335 0.1 0.045 0.519 0.335 0.08 0.045 0.654 0.205 0.08 0.055 0.614 0.215 0.115 0.065 0.609 0.205 0.115 0.075 0.604 0.225 0.115 0.055 0.524 0.23 0.06 0.095 0.219 0.315 0.065 0.075 0.629 0.31 0.095 0.08 0.639 0.325 0.085 0.06 0.219 0.41 0.25 0.11 0.354 0.46 0.185 0.11 0.439 0.515 0.09 0.075 0.359 0.455 0.175 0.125 0.449 0.525 0.08 0.07 0.574 0.46 0.06 0.105 0.579 0.46 0.105 0.1 0.529 0.47 0.15 0.145 0.584 0.475 0.085 0.09 0.354 0.52 0.08 0.06 0.219 0.52 0.115 0.1 0.229 0.53 0.1 0.08 0.229 0.575 0.105 0.045 0.339 0.56 0.085 0.045 0.354 0.535 0.075 0.06 0.299 0.59 0.145 0.05 0.304 0.58 0.12 0.045 0.594 0.555 0.075 0.05 0.534 0.58 0.14 0.06 0.504 0.66 0.07 0.06 0.494 0.73 0.075 0.09 0.504 0.695 0.07 0.095 0.219 0.665 0.075 0.145 0.494 0.755 0.085 0.075 0.704 0.665 0.07 0.21 0.434 0.72 0.055 0.1 0.569 0.695 0.205 0.185 0.219 0.73 0.29 0.13 0.574 0.665 0.08 0.055 0.634 0.665 0.095 0.045 0.499 0.725 0.08 0.135 0.314 0.71 0.155 0.065 0.264 0.72 0.19 0.105 0.264 0.725 0.185 0.095 0.249 0.725 0.12 0.11 0.379 0.77 0.08 0.055 0.509 0.785 0.055 0.06 0.644 0.875 0.13 0.085 0.664 0.875 0.11 0.075 0.329 0.025 0.08 0.115 0.639 0.235 0.135 0.15 0.354 0.46 0.185 0.12 0.354 0.46 0.185 0.135 0.229 0.225 0.08 0.095 0.219 0.72 0.29 0.14 0.569 0.67 0.205 0.21 0.219 0.315 0.1 0.075 0.219 0.23 0.09 0.085 0.219 0.41 0.295 0.11 0.219 0.665 0.27 0.145 0.219 0.225 0.09 0.14 0.294 0.665 0.2 0.05 0.579 0.46 0.105 0.145 0.549 0.46 0.14 0.145 0.219 0.41 0.295 0.125 0.219 0.59 0.11 0.05 0.639 0.235 0.135 0.155 0.629 0.235 0.145 0.155 0.314 0.71 0.155 0.115 0.334 0.56 0.09 0.045 0.264 0.72 0.225 0.1 0.264 0.72 0.225 0.105 0.219 0.71 0.29 0.15 0.249 0.725 0.125 0.11 0.219 0.665 0.27 0.17 0.494 0.73 0.075 0.115 0.494 0.73 0.085 0.115 0.219 0.0 0.14 0.14 0.219 0.07 0.14 0.14 0.219 0.14 0.14 0.14";
                var rois = roiCoordinates.Split(' ').Select(x => float.Parse(x)).ToList();
                roiList.AddRange(rois);
                roiList.AddRange(rois);
                roiList.AddRange(rois);
                roiList.AddRange(rois);
                roiList.AddRange(rois);
                roiList.AddRange(rois);
                roiList.AddRange(rois);
                roiList.AddRange(rois);
                roiList.AddRange(rois);
                roiList.AddRange(rois);
                roiList.AddRange(roiList);
                roiList.AddRange(roiList);

                // Create input data map
                var inputDataMap = new Dictionary<Variable, Value>();
                var inputVal = Value.CreateBatch(modelFunc.Arguments.First().Shape, resizedCHW, GlobalParameters.Device);
                inputDataMap.Add(modelFunc.Arguments.First(), inputVal);

                inputVal = Value.CreateBatch(modelFunc.Arguments[1].Shape, roiList, GlobalParameters.Device);
                inputDataMap.Add(modelFunc.Arguments[1], inputVal);

                Variable outputVar = modelFunc.Outputs.Where(x => (x.Shape.TotalSize == 84000)).ToList()[0];

                // Create output data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);


                // Start evaluation on the device
                modelFunc.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);

                // Get evaluate result as dense output
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);
                Dictionary<int, float> outputPred = new Dictionary<int, float>();
                
                for (int i = 0; i < outputData[0].Count; i++)
                {
                    outputPred.Add(i, outputData[0][i]);
                }

                var topList = outputPred.OrderByDescending(x => (x.Value)).Take(topK).ToList();
                List<PredResult> result = new List<PredResult>();
                float sumpredresult = outputPred.Sum(x => (x.Value));
                float avgpredresult = outputPred.Average(x => (x.Value));
                float min = outputPred.Min(x => (x.Value));
                float max = outputPred.Max(x => (x.Value));

                foreach (var item in topList)
                {
                    result.Add(new PredResult()
                    {
                        Score = item.Value,
                        Name = actualValues[item.Key]
                    });
                }

                Logging.WriteTrace("Prediction Completed");

                return result;
            }
            catch (Exception ex)
            {
                Logging.WriteTrace(ex);
                throw ex;
            }
        }
    }
}
