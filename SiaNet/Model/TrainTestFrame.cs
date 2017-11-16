using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    /// <summary>
    /// The placeholder for split dataset in X and Y
    /// </summary>
    public class XYFrame
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="XYFrame"/> class.
        /// </summary>
        public XYFrame()
        {
            XFrame = new DataFrame();
            YFrame = new DataFrame();
        }

        /// <summary>
        /// Gets or sets the X frame part of the train/test dataset.
        /// </summary>
        /// <value>
        /// The x frame.
        /// </value>
        public DataFrame XFrame { get; set; }

        /// <summary>
        /// Gets or sets the Y frame part of the train/test dataset.
        /// </summary>
        /// <value>
        /// The y frame.
        /// </value>
        public DataFrame YFrame { get; set; }

        /// <summary>
        /// Gets or sets the current batch of the training set.
        /// </summary>
        /// <value>
        /// The current batch.
        /// </value>
        public XYFrame CurrentBatch { get; set; }

        /// <summary>
        /// Splits the current dataset into train and test batch.
        /// </summary>
        /// <param name="testSplitSize">Size of the test split. (value between 0.01 and 0.99)</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException">Please use test split range between 0.01 and 0.99</exception>
        public TrainTestFrame SplitTrainTest(double testSplitSize)
        {
            if (testSplitSize < 0.01 && testSplitSize > 0.99)
            {
                throw new ArgumentException("Please use test split range between 0.01 and 0.99");
            }

            TrainTestFrame result = new TrainTestFrame();
            int totalRows = XFrame.Data.Count;
            int testRows = (int)(totalRows * testSplitSize);
            int trainRows = totalRows - testRows;

            result.Train.XFrame.Data = XFrame.Data.Take(trainRows).ToList();
            result.Train.XFrame.Columns = XFrame.Columns;

            result.Test.XFrame.Data = XFrame.Data.Skip(trainRows).Take(testRows).ToList();
            result.Test.XFrame.Columns = XFrame.Columns;

            result.Train.YFrame.Data = YFrame.Data.Take(trainRows).ToList();
            result.Train.YFrame.Columns = YFrame.Columns;

            result.Test.YFrame.Data = YFrame.Data.Skip(trainRows).Take(testRows).ToList();
            result.Test.YFrame.Columns = YFrame.Columns;

            return result;
        }

        /// <summary>
        /// Gets the next batch of data for training.
        /// </summary>
        /// <param name="current">The current batch number.</param>
        /// <param name="batchSize">Size of the batch.</param>
        /// <returns></returns>
        public bool NextBatch(int current, int batchSize)
        {
            bool next = true;
            int totalRows = XFrame.Data.Count;
            int skipRows = (current - 1) * batchSize;

            if (skipRows < totalRows)
            {
                CurrentBatch = new XYFrame();
                CurrentBatch.XFrame.Data = XFrame.Data.Skip(skipRows).Take(batchSize).ToList();
                CurrentBatch.XFrame.Columns = XFrame.Columns;

                CurrentBatch.YFrame.Data = YFrame.Data.Skip(skipRows).Take(batchSize).ToList();
                CurrentBatch.YFrame.Columns = YFrame.Columns;
            }
            else
            {
                next = false;
                CurrentBatch = null;
            }

            return next;
        }

        /// <summary>
        /// Saves train and test data to the specified folder as compressed binary stream.
        /// </summary>
        /// <param name="folder">The folder.</param>
        public void Save(string folder)
        {
            try
            {
                if(!Directory.Exists(folder))
                {
                    Directory.CreateDirectory(folder);
                }

                XFrame.SaveStream(string.Format("{0}\\train.sia", folder));
                YFrame.SaveStream(string.Format("{0}\\test.sia", folder));
            }
            catch (Exception e)
            {
                throw e;
            }
        }

        /// <summary>
        /// Loads the train and test data from folder.
        /// </summary>
        /// <param name="folder">The folder.</param>
        public void Load(string folder)
        {
            try
            {
                XFrame = DataFrame.LoadStream(string.Format("{0}\\train.sia", folder));
                YFrame = DataFrame.LoadStream(string.Format("{0}\\test.sia", folder));
            }
            catch (Exception e)
            {
                throw e;
            }
        }
    }

    /// <summary>
    /// Placeholder for traing and test batch of data (XYFrame)
    /// </summary>
    public class TrainTestFrame
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TrainTestFrame"/> class.
        /// </summary>
        public TrainTestFrame()
        {
            Train = new XYFrame();
            Test = new XYFrame();
        }

        /// <summary>
        /// Gets or sets the training set of data.
        /// </summary>
        /// <value>
        /// The train.
        /// </value>
        public XYFrame Train { get; set; }

        /// <summary>
        /// Gets or sets the test set of data.
        /// </summary>
        /// <value>
        /// The test.
        /// </value>
        public XYFrame Test { get; set; }

    }
}
