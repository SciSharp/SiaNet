using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    public class XYFrame
    {
        public XYFrame()
        {
            XFrame = new DataFrame();
            YFrame = new DataFrame();
        }

        public DataFrame XFrame { get; set; }

        public DataFrame YFrame { get; set; }

        public XYFrame CurrentBatch { get; set; }

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

    public class TrainTestFrame
    {
        public TrainTestFrame()
        {
            Train = new XYFrame();
            Test = new XYFrame();
        }

        public XYFrame Train { get; set; }

        public XYFrame Test { get; set; }

    }
}
