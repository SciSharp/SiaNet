using CNTK;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    /// <summary>
    /// Enum type for image data generator
    /// </summary>
    public enum ImageGenType
    {
        FromFolder,
        FromTextFile
    }

    /// <summary>
    /// 
    /// </summary>
    public class ImageDataGenerator
    {
        public string FileName { get; set; }

        private string featureStreamName = "features";

        private string labelsStreamName = "labels";

        public MinibatchSource miniBatchSource;

        public ImageGenType GenType { get; set; }

        public MinibatchData CurrentBatchX { get; set; }
        public MinibatchData CurrentBatchY { get; set; }

        private Variable featureVariable;

        private Variable labelVariable;

        public StreamInformation featureStreamInfo;

        public StreamInformation labelStreamInfo;

        public ImageDataGenerator()
        {

        }

        public static ImageDataGenerator FlowFromText(string fileName)
        {
            ImageDataGenerator result = new ImageDataGenerator()
            {
                FileName = fileName,
            };

            result.GenType = ImageGenType.FromTextFile;

            return result;
        }

        internal void LoadTextData(Variable feature, Variable label)
        {
            int imageSize = feature.Shape.Rank == 1 ? feature.Shape[0] : feature.Shape[0] * feature.Shape[1] * feature.Shape[2];
            int numClasses = label.Shape[0];
            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[] { new StreamConfiguration(featureStreamName, imageSize), new StreamConfiguration(labelsStreamName, numClasses) };

            miniBatchSource = MinibatchSource.TextFormatMinibatchSource(FileName, streamConfigurations, MinibatchSource.InfinitelyRepeat);
            featureVariable = feature;
            labelVariable = label;
            featureStreamInfo = miniBatchSource.StreamInfo(featureStreamName);
            labelStreamInfo = miniBatchSource.StreamInfo(labelsStreamName);
        }

        public bool NextBatch(int batchSize)
        {
            var minibatchData = miniBatchSource.GetNextMinibatch((uint)batchSize, GlobalParameters.Device);
            bool result = minibatchData.Values.Any(a => a.sweepEnd);
            if (result == true)
                return result;

            CurrentBatchX = minibatchData[featureStreamInfo];
            CurrentBatchY = minibatchData[labelStreamInfo];
            
            return result;
        }
    }
}
