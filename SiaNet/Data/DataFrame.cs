namespace SiaNet.Data
{
    using System;
    using SiaNet.Engine;
    using System.Linq;
    using NumSharp;

    /// <summary>
    /// Data frame to load data like CSV, images and text in binary format. The instance is then send to Train or Predict method
    /// </summary>
    public class DataFrame
    {
        internal IBackend K = Global.CurrentBackend;

        internal NDArray UnderlayingVariable;

        private Tensor UnderlayingTensor;

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

        public void Load(float[] data, long[] shape)
        {
            UnderlayingVariable = np.array<float>(data);
            UnderlayingVariable.reshape(BackendUtil.CastShapeInt(shape));
        }

        /// <summary>
        /// Gets the underlaying tensor instance.
        /// </summary>
        /// <returns></returns>
        public Tensor GetTensor()
        {
            if (UnderlayingTensor == null)
            {
                UnderlayingTensor = K.CreateVariable(UnderlayingVariable.Data<float>(), Shape);
            }

            return UnderlayingTensor;
        }

        public Array DataArray
        {
            get
            {
                return UnderlayingVariable.Data<float>();
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
                return K.SliceRows(GetTensor(), start, start + size - 1);
                //data = UnderlayingVariable[new NDArray(Enumerable.Range(start, size).ToArray())];
            }
            else
            {
                //int count = (int)Shape[0] - start;
                //data = UnderlayingVariable[new NDArray(Enumerable.Range(start, count).ToArray())];

                return K.SliceRows(GetTensor(), start, Shape[0] - 1);
            }

            //return K.CreateVariable(data.Data<float>(), BackendUtil.Int2Long(data.shape));
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

            if(!string.IsNullOrWhiteSpace(title))
            {
                Console.WriteLine("-----------------{0}----------------", title);
            }

            Console.WriteLine(UnderlayingVariable[new NDArray(Enumerable.Range(0, count - 1).ToArray())].ToString());
        }

        /// <summary>
        /// Round to nearest integer number
        /// </summary>
        public void Round()
        {
        }

        public void Max(int? dim = null)
        {
            if (dim.HasValue)
                UnderlayingVariable = UnderlayingVariable.max(dim.Value);
            else
                UnderlayingVariable = UnderlayingVariable.max<float>();
        }

        public void Min(int? dim = null)
        {
            UnderlayingVariable = UnderlayingVariable.min(dim);
        }

        public void Argmax()
        {
            UnderlayingVariable = UnderlayingVariable.argmax();
        }
    }
}
