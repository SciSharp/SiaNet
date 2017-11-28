namespace SiaNet.Model.Layers
{
    using System;
    using System.Dynamic;

    /// <summary>
    /// Average pooling operation for 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class AvgPool3D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AvgPool3D"/> class.
        /// </summary>
        internal AvgPool3D()
        {
            base.Name = "AvgPool3D";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AvgPool3D"/> class.
        /// </summary>
        /// <param name="poolSize">Tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each dimension.</param>
        /// <param name="strides">Tuple of 3 integers, or None. Strides values.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        public AvgPool3D(Tuple<int, int, int> poolSize, Tuple<int, int, int> strides = null, bool padding = true)
            : this()
        {
            PoolSize = poolSize;
            Strides = strides == null ? Tuple.Create(1, 1, 1) : strides;
            Padding = padding;
        }

        /// <summary>
        /// Tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each dimension.
        /// </summary>
        /// <value>
        /// The size of the pool.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int, int> PoolSize
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
        /// Tuple of 3 integers, or None. Strides values.
        /// </summary>
        /// <value>
        /// The strides.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int, int> Strides
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
