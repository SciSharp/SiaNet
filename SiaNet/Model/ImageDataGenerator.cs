namespace SiaNet.Model
{
    using CNTK;
    using SiaNet.Common;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Enum type for image data generator
    /// </summary>
    public enum ImageGenType
    {
        FromFolder,
        FromTextFile
    }

    /// <summary>
    /// Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches) indefinitely.
    /// </summary>
    public class ImageDataGenerator
    {
        public string FileName { get; set; }

        private string featureStreamName = "features";

        private string labelsStreamName = "labels";

        private MinibatchSource miniBatchSource;
        
        public Value CurrentBatchY { get; set; }

        private Variable featureVariable;

        private Variable labelVariable;

        private StreamInformation featureStreamInfo;

        private StreamInformation labelStreamInfo;

        
        /// <summary>
        /// Gets or sets the type of the image data generator.
        /// </summary>
        /// <value>
        /// The type of the gen.
        /// </value>
        public ImageGenType GenType { get; set; }

        /// <summary>
        /// Gets or sets the current batch X data.
        /// </summary>
        /// <value>
        /// The current batch x.
        /// </value>
        public Value CurrentBatchX { get; set; }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageDataGenerator"/> class.
        /// </summary>
        public ImageDataGenerator()
        {

        }

        /// <summary>
        /// Flows image dataset from text file.
        /// </summary>
        /// <param name="fileName">Name of the file which stores the image dataset information.</param>
        /// <returns></returns>
        public static ImageDataGenerator FlowFromText(string fileName)
        {
            ImageDataGenerator result = new ImageDataGenerator()
            {
                FileName = fileName,
            };

            result.GenType = ImageGenType.FromTextFile;

            return result;
        }

        internal void LoadTextData(CNTK.Variable feature, CNTK.Variable label)
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

        public void LoadSample(SampleDataset sample, CNTK.Variable feature, CNTK.Variable label)
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

        /// <summary>
        /// Gets the next the batch to train using the batch size.
        /// </summary>
        /// <param name="batchSize">Size of the batch.</param>
        /// <returns></returns>
        public bool NextBatch(int batchSize)
        {
            var minibatchData = miniBatchSource.GetNextMinibatch((uint)batchSize, GlobalParameters.Device);
            bool result = minibatchData.Values.Any(a => a.sweepEnd);
            if (result == true)
                return result;

            CurrentBatchX = minibatchData[featureStreamInfo].data;
            CurrentBatchY = minibatchData[labelStreamInfo].data;
            
            return result;
        }
    }
}
