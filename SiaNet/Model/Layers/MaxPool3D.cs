using System;
using CNTK;
using Newtonsoft.Json;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Max pooling operation for 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class MaxPool3D : LayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="MaxPool3D" /> class.
        /// </summary>
        /// <param name="poolSize">
        ///     Tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the
        ///     size of the 3D input in each dimension.
        /// </param>
        /// <param name="strides">Tuple of 3 integers, or None. Strides values.</param>
        /// <param name="padding">
        ///     Boolean, if true results in padding the input such that the output has the same length as the
        ///     original input.
        /// </param>
        public MaxPool3D(Tuple<int, int, int> poolSize, Tuple<int, int, int> strides = null, bool padding = true)
            : this()
        {
            PoolSize = poolSize;
            Strides = strides == null ? Tuple.Create(1, 1, 1) : strides;
            Padding = padding;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="MaxPool3D" /> class.
        /// </summary>
        internal MaxPool3D()
        {
        }

        /// <summary>
        ///     Boolean, if true results in padding the input such that the output has the same length as the original input.
        /// </summary>
        /// <value>
        ///     <c>true</c> if padding; otherwise, <c>false</c>.
        /// </value>
        [JsonIgnore]
        public bool Padding
        {
            get => GetParam<bool>("Padding");

            set => SetParam("Padding", value);
        }

        /// <summary>
        ///     Tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size of the 3D
        ///     input in each dimension.
        /// </summary>
        /// <value>
        ///     The size of the pool.
        /// </value>
        [JsonIgnore]
        public Tuple<int, int, int> PoolSize
        {
            get => GetParam<Tuple<int, int, int>>("PoolSize");

            set => SetParam("PoolSize", value);
        }

        /// <summary>
        ///     Tuple of 3 integers, or None. Strides values.
        /// </summary>
        /// <value>
        ///     The strides.
        /// </value>
        [JsonIgnore]
        public Tuple<int, int, int> Strides
        {
            get => GetParam<Tuple<int, int, int>>("Strides");

            set => SetParam("Strides", value);
        }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Pooling(inputFunction, PoolingType.Max,
                new[] {PoolSize.Item1, PoolSize.Item2, PoolSize.Item3},
                new[] {Strides.Item1, Strides.Item2, Strides.Item3}, new BoolVector(new[] {Padding, Padding, Padding}));
        }
    }
}