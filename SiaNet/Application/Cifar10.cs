using CNTK;
using SiaNet.Common;
using SiaNet.Common.Resources;
using SiaNet.Model;
using System;
using SiaNet.Processing;
using System.Collections.Generic;
using System.Drawing;
using System.Dynamic;
using System.IO;
using System.Linq;
using Function = CNTK.Function;
using Variable = CNTK.Variable;

namespace SiaNet.Application
{
    /// <summary>
    /// Image classification with Cifar 10 models. Refer: https://www.cs.toronto.edu/~kriz/cifar.html
    /// </summary>
    public class Cifar10
    {
        /// <summary>
        /// The cifar 10 model
        /// </summary>
        Cifar10Model model;

        /// <summary>
        /// The model function
        /// </summary>
        Function modelFunc;

        /// <summary>
        /// The actual values
        /// </summary>
        Dictionary<int, string> actualValues;

        /// <summary>
        /// Initializes a new instance of the <see cref="Cifar10"/> class.
        /// </summary>
        /// <param name="model">The cifar 10 model to use for this application.</param>
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

        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <exception cref="Exception">Invalid model selected!</exception>
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

        /// <summary>
        /// Predicts the specified image.
        /// </summary>
        /// <param name="imagePath">The image path.</param>
        /// <param name="topK">The top k accurate result to return.</param>
        /// <returns></returns>
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

        /// <summary>
        /// Predicts the specified image.
        /// </summary>
        /// <param name="imageBytes">The image data in byte array.</param>
        /// <param name="topK">The top k accurate result to return.</param>
        /// <returns></returns>
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

        /// <summary>
        /// Predicts the specified image.
        /// </summary>
        /// <param name="bmp">The image in bitmap format.</param>
        /// <param name="topK">The top k accurate result to return.</param>
        /// <returns></returns>
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
