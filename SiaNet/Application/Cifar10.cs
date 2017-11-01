using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using SiaNet.Processing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet.Model;
using SiaNet.Common.Resources;
using System.Dynamic;
using SiaNet.Common;

namespace SiaNet.Application
{
    public class Cifar10
    {
        Cifar10Model model;

        Function modelFunc;

        Dictionary<int, string> actualValues;
        
        public Cifar10(Cifar10Model model)
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
                    case Cifar10Model.ResNet110:
                        Downloader.DownloadModel(PreTrainedModelPath.Cifar10Path.ResNet110);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.Cifar10Path.ResNet110);
                        break;
                    case Cifar10Model.ResNet20:
                        Downloader.DownloadModel(PreTrainedModelPath.Cifar10Path.ResNet20);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.Cifar10Path.ResNet20);
                        break;
                    default:
                        throw new Exception("Invalid model selected!");
                }

                modelFunc = Function.Load(modelFile, GlobalParameters.Device);
                Logging.WriteTrace("Model loaded.");
            }
            catch(Exception ex)
            {
                Logging.WriteTrace(ex);
                throw ex;
            }
        }

        public List<PredResult> Predict(string imagePath, int topK = 3)
        {
            try
            {
                Bitmap bmp = new Bitmap(Image.FromFile(imagePath));
                return Predict(bmp, topK);
            }
            catch (Exception ex)
            {
                Logging.WriteTrace(ex);
                throw ex;
            }
        }

        public List<PredResult> Predict(byte[] imageBytes, int topK = 3)
        {
            try
            {
                Bitmap bmp = new Bitmap(Image.FromStream(new MemoryStream(imageBytes)));
                return Predict(bmp, topK);
            }
            catch (Exception ex)
            {
                Logging.WriteTrace(ex);
                throw ex;
            }
        }

        public List<PredResult> Predict(Bitmap bmp, int topK = 3)
        {
            try
            {
                Variable inputVar = modelFunc.Arguments.Single();

                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];

                var resized = bmp.Resize(imageWidth, imageHeight, true);
                List<float> resizedCHW = resized.ParallelExtractCHW();
                
                // Create input data map
                var inputDataMap = new Dictionary<Variable, Value>();
                var inputVal = Value.CreateBatch(inputShape, resizedCHW, GlobalParameters.Device);
                inputDataMap.Add(inputVar, inputVal);

                Variable outputVar = modelFunc.Outputs.Where(x => (x.Shape.TotalSize == 10)).ToList()[0];

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
                float min = outputPred.Min(x=>(x.Value));
                float max = outputPred.Max(x=>(x.Value));

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
