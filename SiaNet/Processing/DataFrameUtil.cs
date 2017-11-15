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
            Parallel.ForEach(frame.Data, (record) =>
            {
                batch.AddRange(record);
            });

            Value result = Value.CreateBatch(new int[] { frame.Columns.Count }, batch, GlobalParameters.Device);
            return result;
        }
    }
}
