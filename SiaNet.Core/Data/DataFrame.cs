namespace SiaNet.Data
{
    using System;
    using System.Collections.Generic;
    using CNTK;

    /// <summary>
    /// Generic Dataframe class to build your own custom data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Data.IDataFrame" />
    public class DataFrame<T> : IDataFrame<T>
    {
        protected readonly List<T> DataList = new List<T>();

        protected DataFrame()
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataFrame"/> class.
        /// </summary>
        /// <param name="shape">The shape of the dataframe.</param>
        public DataFrame(Shape shape)
        {
            DataShape = shape;
        }

        /// <summary>
        /// Gets the data.
        /// </summary>
        /// <value>
        /// The data.
        /// </value>
        public IEnumerable<T[]> Data
        {
            get
            {
                for (var i = 0; i < DataList.Count; i += DataShape.TotalSize)
                {
                    var data = new T[DataShape.TotalSize];
                    DataList.CopyTo(i, data, 0, data.Length);

                    yield return data;
                }
            }
        }

        /// <summary>
        /// Gets the length.
        /// </summary>
        /// <value>
        /// The length.
        /// </value>
        /// <inheritdoc />
        public int Length {
            get => DataList.Count / DataShape.TotalSize;
        }

        /// <summary>
        /// Gets or sets the shape of the data.
        /// </summary>
        /// <value>
        /// The data shape.
        /// </value>
        /// <inheritdoc />
        public Shape DataShape { get; protected set; }

        /// <summary>
        /// Reshapes the dataframe to new shape.
        /// </summary>
        /// <param name="newShape">The new shape.</param>
        /// <exception cref="System.ArgumentException">The size of the new shape should be identical to the last shape. - newShape</exception>
        /// <inheritdoc />
        public void Reshape(Shape newShape)
        {
            if (DataShape.TotalSize != newShape.TotalSize)
            {
                throw new ArgumentException("The size of the new shape should be identical to the last shape.",
                    nameof(newShape));
            }

            DataShape = newShape;
        }

        /// <summary>
        /// Convert To the CNTK value object.
        /// </summary>
        /// <returns>The CNTK Value object</returns>
        /// <inheritdoc />
        public Value ToValue()
        {
            return Value.CreateBatch(DataShape, DataList, GlobalParameters.Device);
        }

        /// <summary>
        /// Gets or sets the <see cref="System.Single[]"/> at the specified index.
        /// </summary>
        /// <value>
        /// The <see cref="System.Single[]"/>.
        /// </value>
        /// <param name="index">The index.</param>
        /// <returns></returns>
        /// <inheritdoc />
        public T[] this[int index]
        {
            get
            {
                index = index * DataShape.TotalSize;
                var data = new T[DataShape.TotalSize];
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

        /// <summary>
        /// Adds the specified data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <exception cref="System.ArgumentException">The size of the data must be identical to the size of data shape. - data</exception>
        public void Add(params T[] data)
        {
            if (data.Length != DataShape.TotalSize)
            {
                throw new ArgumentException("The size of the data must be identical to the size of data shape.",
                    nameof(data));
            }

            DataList.AddRange(data);
        }

        public static DataFrame<T> LoadBinary(string path, int rows, int columns)
        {
            DataFrame<T> dataFrame = new DataFrame<T>(new Shape(columns));

            //var buffer = new byte[System.Runtime.InteropServices.Marshal.SizeOf<T>() * rows * columns];
            byte[] buffer;
            using (var reader = new System.IO.BinaryReader(System.IO.File.OpenRead(path)))
            {
                //reader.Read(buffer, 0, buffer.Length);
                buffer = reader.ReadBytes(rows * columns);
            }

            for (int i = 0; i < rows; i++)
            {
                var dst = new T[columns];
                System.Buffer.BlockCopy(buffer, i * columns, dst, 0, columns);
                dataFrame.DataList.AddRange(dst);
            }

            return dataFrame;
        }
    }
}