using CNTK;
using SiaNet.Model;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Processing
{
    internal class DataFrameUtil
    {
        internal static Value GetValueBatch(DataFrame frame)
        {
            List<float> batch = new List<float>();
            frame.Data.ForEach((record) =>
            {
                batch.AddRange(record);
            });

            Value result = null;

            if (frame.FrameType == FrameType.IMG)
            {
                result = Value.CreateBatch(new int[] { frame.imageDimension.Item1, frame.imageDimension.Item2, frame.imageDimension.Item3 }, batch, GlobalParameters.Device);
            }
            else if (frame.FrameType == FrameType.CSV)
            {
                result = Value.CreateBatch(new int[] { frame.Columns.Count }, batch, GlobalParameters.Device);
            }

            return result;
        }
    }
}
