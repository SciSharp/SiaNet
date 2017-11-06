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
            DataTable dt = frame.Frame.ToTable();
            int dim = dt.Columns.Count;
            List<float> batch = new List<float>();
            foreach (DataRow item in dt.Rows)
            {
                foreach (var row in item.ItemArray)
                {
                    if (row != null)
                    {
                        batch.Add((float)row);
                    }
                    else
                    {
                        batch.Add(0);
                    }
                }
            }

            Value result = Value.CreateBatch(new int[] { dim }, batch, GlobalParameters.Device);
            return result;
        }
    }
}
