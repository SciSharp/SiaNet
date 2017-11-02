using CNTK;
using Emgu.CV;
using Emgu.CV.Structure;
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

        public List<PredResult> Predict(string imagePath)
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

        public List<PredResult> Predict(byte[] imageBytes)
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

        public List<PredResult> Predict(Bitmap bmp)
        {
            try
            {
                var resized = bmp.Resize(1000, 1000, true);
                List<float> roiList = GenerateROIS(resized, model);
                
                List<float> resizedCHW = resized.ParallelExtractCHW();
                
                // Create input data map
                var inputDataMap = new Dictionary<Variable, Value>();
                var inputVal = Value.CreateBatch(modelFunc.Arguments.First().Shape, resizedCHW, GlobalParameters.Device);
                inputDataMap.Add(modelFunc.Arguments.First(), inputVal);

                inputVal = Value.CreateBatch(modelFunc.Arguments[1].Shape, roiList, GlobalParameters.Device);
                inputDataMap.Add(modelFunc.Arguments[1], null);

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
                Dictionary<int, float> outputPred = new Dictionary<int, float>();

                for (int i = 0; i < outputData[0].Count; i++)
                {
                    outputPred.Add(i, outputData[0][i]);
                }

                List<PredResult> result = new List<PredResult>();

                Logging.WriteTrace("Prediction Completed");
                
                return null;
            }
            catch (Exception ex)
            {
                Logging.WriteTrace(ex);
                throw ex;
            }
        }

        public List<float> GenerateROIS(Bitmap bmp, FastRCNNModel model)
        {
            int selectRois = 16000;
            int selectionParam = 250;
            if(model == FastRCNNModel.Grocery100)
            {
                selectRois = 400;
                selectionParam = 1000;
            }

            float[] roiList = new float[selectRois * 4];
            
            Emgu.CV.Image<Bgr, byte> img = new Image<Bgr, byte>(bmp);
            Emgu.CV.XImgproc.SelectiveSearchSegmentation seg = new Emgu.CV.XImgproc.SelectiveSearchSegmentation();
            var resizedImg = img.Resize(200, 200, Emgu.CV.CvEnum.Inter.Nearest);
            seg.SetBaseImage(resizedImg);
            seg.SwitchToSelectiveSearchQuality(selectionParam, selectionParam);
            var rects = seg.Process();
            rects = rects.OrderBy(x => (x.X)).ToArray();
            int counter = 0;
            foreach (var item in rects)
            {
                roiList[counter] = item.X * 5; counter++;
                roiList[counter] = item.Y * 5; counter++;
                roiList[counter] = item.Width * 5; counter++;
                roiList[counter] = item.Height * 5; counter++;
            }

            if(counter <= selectRois)
            {
                for(int i = counter; i< selectRois; i++)
                {
                    roiList[i] = 0;
                }
            }

            return roiList.ToList();
        }

        public Variable GetOutputVar(FastRCNNModel model)
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
