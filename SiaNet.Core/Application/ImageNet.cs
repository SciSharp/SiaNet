using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet;
using SiaNet.Common.Resources;
using System.Dynamic;
using SiaNet.Common;
using Function = CNTK.Function;
using Variable = CNTK.Variable;
using SiaNet.Utils;

namespace SiaNet.Application
{
    /// <summary>
    /// Image classification application using ImageNet 1K model
    /// </summary>
    public class ImageNet
    {
        /// <summary>
        /// The model
        /// </summary>
        ImageNetModel model;

        /// <summary>
        /// The model function
        /// </summary>
        Function modelFunc;

        /// <summary>
        /// The actual values
        /// </summary>
        Dictionary<int, string> actualValues;

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageNet"/> class.
        /// </summary>
        /// <param name="model">The model to use. <see cref="ImageNetModel"/></param>
        public ImageNet(ImageNetModel model)
        {
            this.model = model;
            actualValues = new Dictionary<int, string>();
            ExpandoObject jsonValues = Newtonsoft.Json.JsonConvert.DeserializeObject<ExpandoObject>(PredMap.ImageNet1K);
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
                    case ImageNetModel.AlexNet:
                        Downloader.DownloadModel(PreTrainedModelPath.ImageNetPath.AlexNet);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.ImageNetPath.AlexNet);
                        break;
                    case ImageNetModel.InceptionV3:
                        Downloader.DownloadModel(PreTrainedModelPath.ImageNetPath.InceptionV3);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.ImageNetPath.InceptionV3);
                        break;
                    case ImageNetModel.ResNet18:
                        Downloader.DownloadModel(PreTrainedModelPath.ImageNetPath.ResNet18);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.ImageNetPath.ResNet18);
                        break;
                    case ImageNetModel.ResNet34:
                        Downloader.DownloadModel(PreTrainedModelPath.ImageNetPath.ResNet34);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.ImageNetPath.ResNet34);
                        break;
                    case ImageNetModel.ResNet50:
                        Downloader.DownloadModel(PreTrainedModelPath.ImageNetPath.ResNet50);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.ImageNetPath.ResNet50);
                        break;
                    case ImageNetModel.ResNet101:
                        Downloader.DownloadModel(PreTrainedModelPath.ImageNetPath.ResNet101);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.ImageNetPath.ResNet101);
                        break;
                    case ImageNetModel.ResNet152:
                        Downloader.DownloadModel(PreTrainedModelPath.ImageNetPath.ResNet152);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.ImageNetPath.ResNet152);
                        break;
                    case ImageNetModel.VGG16:
                        Downloader.DownloadModel(PreTrainedModelPath.ImageNetPath.VGG16);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.ImageNetPath.VGG16);
                        break;
                    case ImageNetModel.VGG19:
                        Downloader.DownloadModel(PreTrainedModelPath.ImageNetPath.VGG19);
                        modelFile = baseFolder + "\\" + Path.GetFileName(PreTrainedModelPath.ImageNetPath.VGG19);
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
        /// Predicts the specified image path.
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
        /// Predicts the specified image bytes.
        /// </summary>
        /// <param name="imageBytes">The image in byte arrary.</param>
        /// <param name="topK">The top k accurate result to return.</param>
        /// <returns>
        /// Prediction result with bounded box
        /// </returns>
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
        /// Predicts the specified BMP.
        /// </summary>
        /// <param name="bmp">The image in bitmap format.</param>
        /// <param name="topK">The top k accurate result to return.</param>
        /// <returns></returns>
        public List<PredResult> Predict(Bitmap bmp, int topK = 3)
        {
            try
            {
                Variable inputVar = modelFunc.Arguments[0];

                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];

                var resized = bmp.Resize(imageWidth, imageHeight, true);
                List<float> resizedCHW = resized.ParallelExtractCHW();
                
                // Create input data map
                var inputDataMap = new Dictionary<Variable, Value>();
                var inputVal = Value.CreateBatch(inputShape, resizedCHW, GlobalParameters.Device);
                inputDataMap.Add(inputVar, inputVal);
                inputVar = modelFunc.Arguments[1];
                //inputDataMap.Add(inputVar, null);

                Variable outputVar = modelFunc.Outputs.Where(x => (x.Shape.TotalSize == 1000)).ToList()[0];

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
