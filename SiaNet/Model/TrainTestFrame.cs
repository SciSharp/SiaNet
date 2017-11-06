using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
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
            DataTable dt_x = XFrame.Frame.ToTable();
            DataTable dt_y = YFrame.Frame.ToTable();
            int totalRows = dt_x.Rows.Count;
            int testRows = (int)(totalRows * testSplitSize);
            int trainRows = totalRows - testRows;
            
            result.Train.XFrame.Frame = dt_x.Select().Take(trainRows).CopyToDataTable().AsDataView();
            result.Test.XFrame.Frame = dt_x.Select().Skip(trainRows).Take(testRows).CopyToDataTable().AsDataView();
            
            result.Train.YFrame.Frame = dt_y.Select().Take(trainRows).CopyToDataTable().AsDataView();
            result.Test.YFrame.Frame = dt_y.Select().Skip(trainRows).Take(testRows).CopyToDataTable().AsDataView();

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
            DataTable dt_x = XFrame.Frame.ToTable();
            DataTable dt_y = YFrame.Frame.ToTable();
            int totalRows = dt_x.Rows.Count;
            int skipRows = (current - 1) * batchSize;

            if (skipRows < totalRows)
            {
                CurrentBatch = new XYFrame();
                CurrentBatch.XFrame.Frame = dt_x.Select().Skip(skipRows).Take(batchSize).CopyToDataTable().AsDataView();
                CurrentBatch.YFrame.Frame = dt_y.Select().Skip(skipRows).Take(batchSize).CopyToDataTable().AsDataView();
            }
            else
            {
                next = false;
                CurrentBatch = null;
            }

            return next;
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
