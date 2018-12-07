using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.IO;
using CsvHelper;

namespace SiaNet.Data
{
    public class DataFrame
    {
        public List<float> DataList = new List<float>();

        internal uint _cols = 0;

        internal NDArray variable;

        internal DataFrame()
        {

        }

        public DataFrame(uint cols)
            :base()
        {
            if (cols == 0)
                throw new ArgumentException("0 columns in 2D array is not acceptable");

            _cols = cols;
        }

        public void AddData(params float[] data)
        {
            DataList.AddRange(data);
        }

        internal void GenerateVariable()
        {
            if (DataList.Count == 0)
                throw new Exception("No data to generate variable. Please add data using AddData method");

            uint rows = (uint)DataList.Count / _cols;
            if (_cols == 1)
            {
                variable = new NDArray(DataList.ToArray(), new Shape(rows));
            }
            else
            {
                variable = new NDArray(DataList.ToArray(), new Shape(rows, _cols));
            }
        }

        public NDArray ToVariable()
        {
            GenerateVariable();
            return variable;
        }

        public NDArray this[int start, int? end]
        {
            get
            {
                return variable.SliceAxis(1, start, end);
            }
        }
    }
}
