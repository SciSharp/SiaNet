using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Data
{
    public class DataFrameIter : DataIter
    {
        private NDArray _data;
        private NDArray _label;
        private int cursor = 0;
        private uint num_data;

        public DataFrameIter(NDArray data, NDArray label)
        {
            _data = data;
            _label = label;
            BatchSize = 32;
            num_data = data.GetShape()[0];
            cursor = (int)-BatchSize;
        }

        public override void BeforeFirst()
        {
            cursor = (int)-BatchSize;
        }

        public override NDArray GetData()
        {
            uint start = (uint)cursor;
            uint end = (uint)cursor + BatchSize;
            if(end >= num_data)
            {
                end = num_data;
            }

            return _data.Slice(start, end);
        }

        public override int[] GetIndex()
        {
            uint start = (uint)cursor;
            uint end = (uint)cursor + BatchSize;
            if (end >= num_data)
            {
                end = num_data;
            }

            List<int> idx = new List<int>();
            for (int i = (int)start; i < end; i++)
            {
                idx.Add((int)i);
            }

            return idx.ToArray();
        }

        public override NDArray GetLabel()
        {
            uint start = (uint)cursor;
            uint end = (uint)cursor + BatchSize;
            if (end >= num_data)
            {
                end = num_data;
            }

            return _label.Slice(start, end).Reshape(new Shape(BatchSize));
        }

        public override int GetPadNum()
        {
            return 0;
        }

        public override bool Next()
        {
            cursor += (int)BatchSize;
            if(cursor < num_data)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
