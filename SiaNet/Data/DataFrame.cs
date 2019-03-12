namespace SiaNet.Data
{
    using System;
    using SiaNet.Engine;

    /// <summary>
    /// Data frame to load data like CSV, images and text in binary format. The instance is then send to Train or Predict method
    /// </summary>
    public class DataFrame
    {
        internal IBackend K = Global.CurrentBackend;

        internal Tensor UnderlayingTensor;

        /// <summary>
        /// Gets the shape of the data frame.
        /// </summary>
        /// <value>
        /// The shape.
        /// </value>
        public long[] Shape
        {
            get
            {
                return UnderlayingTensor.Shape;
            }
        }

        /// <summary>
        /// Reshapes the data frame to specified new shape.
        /// </summary>
        /// <param name="newShape">The new shape.</param>
        public void Reshape(params long[] newShape)
        {
            UnderlayingTensor = UnderlayingTensor.Reshape(Shape);
        }

        /// <summary>
        /// Gets the underlaying tensor instance.
        /// </summary>
        /// <returns></returns>
        public Tensor GetTensor()
        {
            return UnderlayingTensor;
        }

        /// <summary>
        /// Converts to tensor to a data frame.
        /// </summary>
        /// <param name="t">The t.</param>
        public virtual void ToFrame(Tensor t)
        {
            UnderlayingTensor = t;
        }

        /// <summary>
        /// Gets the <see cref="DataFrame"/> with the specified start and end index.
        /// </summary>
        /// <value>
        /// The <see cref="DataFrame"/>.
        /// </value>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException">End must be greater than start</exception>
        public DataFrame this[uint start, uint end]
        {
            get
            {
                if (end <= start)
                {
                    throw new ArgumentException("End must be greater than start");
                }

                DataFrame frame = new DataFrame();
                
                var count = end - start + 1;
                if (count > 0)
                    frame.ToFrame(UnderlayingTensor.SliceCols(start, end));

                return frame;
            }
        }

        /// <summary>
        /// Gets the <see cref="DataFrame"/> at the specified index.
        /// </summary>
        /// <value>
        /// The <see cref="DataFrame"/>.
        /// </value>
        /// <param name="index">The index.</param>
        /// <returns></returns>
        public DataFrame this[uint index]
        {
            get
            {
                DataFrame frame = new DataFrame();
                frame.ToFrame(UnderlayingTensor.SliceCols(index, index));

                return frame;
            }
        }

        /// <summary>
        /// Get batch data with specified start index and size
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="size">The size.</param>
        /// <param name="axis">The axis.</param>
        /// <returns></returns>
        public Tensor GetBatch(int start, int size, int axis = 0)
        {
            if (start + size <= Shape[0])
            {
                return UnderlayingTensor.SliceRows(start, start + size - 1);
            }
            else
            {
                return UnderlayingTensor.SliceRows(start, Shape[0] - 1);
            }
        }

        /// <summary>
        /// Prints the dataframe, by default first 5 records are printed. Helpful to understand the data structure.
        /// </summary>
        /// <param name="count">The count of records to print.</param>
        /// <param name="title">The title to display.</param>
        public void Head(uint count = 5, string title = "")
        {
            K.Print(UnderlayingTensor.SliceRows(0, count), title);
        }
    }
}
