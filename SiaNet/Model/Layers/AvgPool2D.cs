namespace SiaNet.Model.Layers
{
    using System;
    using System.Dynamic;

    /// <summary>
    /// Average pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class AvgPool2D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AvgPool2D"/> class.
        /// </summary>
        public AvgPool2D()
        {
            base.Name = "AvgPool2D";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AvgPool2D"/> class.
        /// </summary>
        /// <param name="poolSize">A tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.</param>
        /// <param name="strides">Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        public AvgPool2D(Tuple<int, int> poolSize, Tuple<int, int> strides = null, bool padding = true)
            : this()
        {
            PoolSize = poolSize;
            Strides = strides == null ? Tuple.Create(1, 1) : strides;
            Padding = padding;
        }

        /// <summary>
        ///  integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.
        /// </summary>
        /// <value>
        /// The size of the pool.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int> PoolSize
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
        /// Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.
        /// </summary>
        /// <value>
        /// The strides.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int> Strides
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
