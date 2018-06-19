using System;
using System.Collections.Generic;
using CNTK;

namespace SiaNet.Model.Data
{
    public class DataFrame : IDataFrame
    {
        protected readonly List<float> DataList = new List<float>();

        public DataFrame(Shape shape)
        {
            DataShape = shape;
        }


        /// <inheritdoc />
        public IEnumerable<float[]> Data
        {
            get
            {
                for (var i = 0; i < DataList.Count; i += DataShape.TotalSize)
                {
                    var data = new float[DataShape.TotalSize];
                    DataList.CopyTo(i, data, 0, data.Length);

                    yield return data;
                }
            }
        }

        /// <inheritdoc />
        public int Length {
            get => DataList.Count / DataShape.TotalSize;
        }

        /// <inheritdoc />
        public Shape DataShape { get; protected set; }

        /// <inheritdoc />
        public void Reshape(Shape newShape)
        {
            if (DataShape.TotalSize != newShape.TotalSize)
            {
                throw new ArgumentException("The size of the new shape should be identical to the last shape.",
                    nameof(newShape));
            }

            DataShape = new Shape();
        }

        /// <inheritdoc />
        public Value ToValue()
        {
            return Value.CreateBatch(DataShape, DataList, GlobalParameters.Device);
        }

        /// <inheritdoc />
        public float[] this[int index]
        {
            get
            {
                index = index * DataShape.TotalSize;
                var data = new float[DataShape.TotalSize];
                DataList.CopyTo(index, data, 0, data.Length);

                return data;
            }
            set
            {
                index = index * DataShape.TotalSize;
                for (int i = 0; i < DataShape.TotalSize; i++)
                {
                    DataList[i + index] = value[i];
                }
            }
        }

        public void Add(float[] data)
        {
            if (data.Length != DataShape.TotalSize)
            {
                throw new ArgumentException("The size of the data must be identical to the size of data shape.",
                    nameof(data));
            }

            DataList.AddRange(data);
        }
    }
}