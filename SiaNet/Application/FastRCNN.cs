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

namespace SiaNet.Application
{
    /// <summary>
    /// Real time object detection application using Fast RCNN models
    /// </summary>
    public class FastRCNN
    {
        /// <summary>
        /// The model
        /// </summary>
        FastRCNNModel model;

        /// <summary>
        /// The model function
        /// </summary>
        Function modelFunc;

        /// <summary>
        /// The actual values
        /// </summary>
        Dictionary<int, string> actualValues;

        /// <summary>
        /// The proposed regions to be detected
        /// </summary>
        List<Rectangle> proposedBoxes;


        /// <summary>
        /// Initializes a new instance of the <see cref="FastRCNN"/> class.
        /// </summary>
        /// <param name="model">The model to use.</param>
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

        /// <summary>
        /// Predicts the specified image.
        /// </summary>
        /// <param name="imagePath">The image path.</param>
        /// <param name="confidence">The confidence level for the prediction result.</param>
        /// <returns></returns>
        public List<PredResult> Predict(string imagePath, double confidence = 0.5)
        {
            try
            {
                Bitmap bmp = new Bitmap(Image.FromFile(imagePath));
                return Predict(bmp);
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
        /// <param name="imageBytes">The image in byte aray.</param>
        /// <param name="confidence">The confidence level for the prediction result.</param>
        /// <returns></returns>
        public List<PredResult> Predict(byte[] imageBytes, double confidence = 0.5)
        {
            try
            {
                Bitmap bmp = new Bitmap(Image.FromStream(new MemoryStream(imageBytes)));
                return Predict(bmp);
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
        /// <param name="confidence">The confidence level for the prediction result.</param>
        /// <returns></returns>
        public List<PredResult> Predict(Bitmap bmp, double confidence = 0.5)
        {
            try
            {
                proposedBoxes = new List<Rectangle>();
                
                var resized = bmp.Resize(1000, 1000, true);
                List<float> roiList = GenerateROIS(resized, model);
                List<float> resizedCHW = resized.ParallelExtractCHW();

                //CalculateROI(resizedCHW);
                // Create input data map
                var inputDataMap = new Dictionary<Variable, Value>();
                var inputVal1 = Value.CreateBatch(modelFunc.Arguments.First().Shape, resizedCHW, GlobalParameters.Device);
                inputDataMap.Add(modelFunc.Arguments.First(), inputVal1);

                var inputVal2 = Value.CreateBatch(modelFunc.Arguments[1].Shape, roiList, GlobalParameters.Device);
                inputDataMap.Add(modelFunc.Arguments[1], inputVal2);

                Variable outputVar = GetOutputVar(model);

                // Create output data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Start evaluation on the device
                modelFunc.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);

                // Get evaluate result as dense output                   
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);
                List<PredResult> result = new List<PredResult>();
                
                var labels = GetLabels(model);
                int numLabels = labels.Length;
                int numRois = outputData[0].Count / numLabels;

                int numBackgroundRois = 0;
                for (int i = 0; i < numRois; i++)
                {
                    var outputForRoi = outputData[0].Skip(i * numLabels).Take(numLabels).ToList();

                    // Retrieve the predicted label as the argmax over all predictions for the current ROI
                    var max = outputForRoi.IndexOf(outputForRoi.Max());

                    if (max > 0)
                    {
                        result.Add(new PredResult()
                        {
                            Name = labels[max],
                            BBox = proposedBoxes[i],
                            Score = outputForRoi.Max()
                        });

                        //Console.WriteLine("Outcome for ROI {0}: {1} \t({2})", i, max, labels[max]);
                    }
                    else
                    {
                        numBackgroundRois++;
                    }
                }

                var groupBoxes = result.GroupBy(x => (x.Name)).ToList();
                result = new List<PredResult>();
                foreach (var item in groupBoxes)
                {
                    int counter = 0;
                    Rectangle unionRect = new Rectangle();
                    
                    foreach (var rect in item.ToList())
                    {
                        if(counter == 0)
                        {
                            unionRect = rect.BBox;
                            continue;
                        }

                        unionRect = Rectangle.Union(unionRect, rect.BBox);
                    }

                    //var orderedList = item.ToList().OrderByDescending(x => (x.BBox.Width * x.BBox.Height)).ToList();
                    foreach (var rect in item.ToList())
                    {
                        unionRect = Rectangle.Intersect(unionRect, rect.BBox);
                    }
                    
                    var goodPred = item.ToList().OrderByDescending(x=>(x.Score)).ToList()[0];
                    goodPred.BBox = unionRect;
                    result.Add(goodPred);
                }

                //foreach (var item in result)
                //{
                //    img.Draw(item.BBox, new Bgr(0, 255, 0));
                //}

                //img.Save("objdet_pred.jpg");

                Logging.WriteTrace("Prediction Completed");
                
                return result;
            }
            catch (Exception ex)
            {
                Logging.WriteTrace(ex);
                throw ex;
            }
        }


        /// <summary>
        /// Gets the labels.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        private string[] GetLabels(FastRCNNModel model)
        {
            string[] result = null;
            switch (model)
            {
                case FastRCNNModel.Grocery100:
                    result =  new[] { "__background__", "avocado", "orange", "butter", "champagne", "egg_Box", "gerkin", "yogurt", "ketchup", "orange_juice", "onion", "pepper", "tomato", "water", "milk", "tabasco", "mustard" };
                    break;
                case FastRCNNModel.Pascal:
                    result =  new[] { "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining_table", "dog", "horse", "motor_bike", "person", "potted_plant", "sheep", "sofa", "train", "tv_monitor" };
                    break;
                default:
                    break;
            }

            return result;
        }

        /// <summary>
        /// Generates the rois.
        /// </summary>
        /// <param name="bmp">The BMP.</param>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        private List<float> GenerateROIS(Bitmap bmp, FastRCNNModel model)
        {
            int selectRois = 4000;
            int selectionParam = 100;
            if(model == FastRCNNModel.Grocery100)
            {
                selectRois = 100;
                selectionParam = 1500;
            }

            float[] roiList = new float[selectRois * 4];

            List<float> inArg3 = new List<float>();
            var imgArray = bmp.ToByteArray();
            var img = DlibDotNet.Dlib.LoadImageData<DlibDotNet.RgbAlphaPixel>(imgArray, 1000, 1000, 1);
            var rects = DlibDotNet.Dlib.FindCandidateObjectLocations(img);
            if (!img.IsDisposed)
            {
                img.Dispose();
            }

            int counter = 0;
            foreach (var item in rects)
            {
                if (counter >= selectRois * 4)
                    break;

                roiList[counter] = item.Top * 4; counter++;
                roiList[counter] = item.Left * 4; counter++;
                roiList[counter] = item.Width * 4; counter++;
                roiList[counter] = item.Height * 4; counter++;

                proposedBoxes.Add(new Rectangle(item.Left * 4, item.Top * 4, (int)item.Width * 4, (int)item.Height * 4));
            }

            if(counter <= selectRois)
            {
                for(int i = counter; i< selectRois; i++)
                {
                    roiList[i] = -1;
                }
            }

            return roiList.ToList();
        }

        /// <summary>
        /// Gets the output variable.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        private Variable GetOutputVar(FastRCNNModel model)
        {
            switch (model)
            {
                case FastRCNNModel.Pascal:
                    return modelFunc.Outputs.Where(x => (x.Shape.TotalSize == 84000)).ToList()[0];
                case FastRCNNModel.Grocery100:
                    return modelFunc.Outputs.Where(x => (x.Shape.TotalSize == 1700)).ToList()[0];
                default:
                    break;
            }

            return null;
        }
    }
}
