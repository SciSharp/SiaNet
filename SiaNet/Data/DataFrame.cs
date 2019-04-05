namespace SiaNet.Data
{
    using System;
    using SiaNet.Engine;
    using System.Linq;
    using NumSharp.Core;

    /// <summary>
    /// Data frame to load data like CSV, images and text in binary format. The instance is then send to Train or Predict method
    /// </summary>
    public class DataFrame
    {
        internal IBackend K = Global.CurrentBackend;

        internal NDArray UnderlayingVariable;

        public DataFrame()
        {

        }

        internal DataFrame(NDArray t)
        {
            UnderlayingVariable = t;
        }

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
                return BackendUtil.Int2Long(UnderlayingVariable.shape);
            }
        }

        /// <summary>
        /// Reshapes the data frame to specified new shape.
        /// </summary>
        /// <param name="newShape">The new shape.</param>
        public void Reshape(params long[] newShape)
        {
            UnderlayingVariable = UnderlayingVariable.reshape(new Shape(BackendUtil.CastShapeInt(newShape)));
        }

        public virtual void Load(params float[] data)
        {
            UnderlayingVariable = np.array<float>(data);
        }

        /// <summary>
        /// Gets the underlaying tensor instance.
        /// </summary>
        /// <returns></returns>
        public Tensor GetTensor()
        {
            return K.CreateVariable(UnderlayingVariable.Data<float>(), Shape);
        }

        public Array DataArray
        {
            get
            {
                return UnderlayingVariable.Data();
            }
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
        public DataFrame this[int start, int end]
        {
            get
            {
                if (end <= start)
                {
                    throw new ArgumentException("End must be greater than start");
                }

                var count = end - start + 1;
                var u = UnderlayingVariable.transpose();
                if (count > 0)
                    return new DataFrame(u[new NDArray(Enumerable.Range(start, count).ToArray())].transpose());

                return null;
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
        public DataFrame this[int index]
        {
            get
            {
                var u = UnderlayingVariable.transpose();
                var result = new DataFrame(u[new NDArray(new int[] { index })].transpose());
                return result;
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
            NDArray data = null;
            
            if (start + size <= Shape[0])
            {
                data = UnderlayingVariable[new NDArray(Enumerable.Range(start, size).ToArray())];
            }
            else
            {
                int count = (int)Shape[0] - start;
                data = UnderlayingVariable[new NDArray(Enumerable.Range(start, count).ToArray())];
            }

            return K.CreateVariable(data.Data<float>(), BackendUtil.Int2Long(data.shape));
        }

        /// <summary>
        /// Prints the dataframe, by default first 5 records are printed. Helpful to understand the data structure.
        /// </summary>
        /// <param name="count">The count of records to print.</param>
        /// <param name="title">The title to display.</param>
        public void Head(int count = 5, string title = "")
        {
            if (count > UnderlayingVariable.shape[0])
            {
                count = UnderlayingVariable.shape[0];
            }

            Console.WriteLine(UnderlayingVariable[new NDArray(new int[] { 0, count })].ToString());
        }
    }
}
