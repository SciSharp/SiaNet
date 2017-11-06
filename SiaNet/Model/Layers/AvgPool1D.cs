namespace SiaNet.Model.Layers
{
    using System.Dynamic;

    /// <summary>
    /// Average pooling for temporal data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class AvgPool1D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AvgPool1D"/> class.
        /// </summary>
        public AvgPool1D()
        {
            base.Name = "AvgPool1D";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AvgPool1D"/> class.
        /// </summary>
        /// <param name="poolSize">Integer, size of the max pooling windows.</param>
        /// <param name="strides">Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        public AvgPool1D(int poolSize, int strides = 1, bool padding = true)
            : this()
        {
            PoolSize = poolSize;
            Strides = strides;
            Padding = padding;
        }

        /// <summary>
        /// Integer, size of the max pooling windows.
        /// </summary>
        /// <value>
        /// The size of the pool.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int PoolSize
        {
            get
            {
                return base.Params.PoolSize;
            }

            set
            {
                base.Params.PoolSize = value;
            }
        }

        /// <summary>
        /// Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.
        /// </summary>
        /// <value>
        /// The strides.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int Strides
        {
            get
            {
                return base.Params.Strides;
            }

            set
            {
                base.Params.Strides = value;
            }
        }

        /// <summary>
        /// Boolean, if true results in padding the input such that the output has the same length as the original input.
        /// </summary>
        /// <value>
        ///   <c>true</c> if padding; otherwise, <c>false</c>.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public bool Padding
        {
            get
            {
                return base.Params.Padding;
            }

            set
            {
                base.Params.Padding = value;
            }
        }
    }
}
