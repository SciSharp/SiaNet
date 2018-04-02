using System;
using Newtonsoft.Json;
using SiaNet.NN;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Max pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class MaxPool2D : LayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="MaxPool2D" /> class.
        /// </summary>
        /// <param name="poolSize">
        ///     A tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve
        ///     the input in both spatial dimension. If only one integer is specified, the same window length will be used for both
        ///     dimensions.
        /// </param>
        /// <param name="strides">Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.</param>
        /// <param name="padding">
        ///     Boolean, if true results in padding the input such that the output has the same length as the
        ///     original input.
        /// </param>
        public MaxPool2D(Tuple<int, int> poolSize, Tuple<int, int> strides = null, bool padding = true)
        {
            PoolSize = poolSize;
            Strides = strides ?? Tuple.Create(1, 1);
            Padding = padding;
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
        ///     integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input
        ///     in both spatial dimension. If only one integer is specified, the same window length will be used for both
        ///     dimensions.
        /// </summary>
        /// <value>
        ///     The size of the pool.
        /// </value>
        [JsonIgnore]
        public Tuple<int, int> PoolSize
        {
            get => GetParam<Tuple<int, int>>("PoolSize");

            set => SetParam("PoolSize", value);
        }

        /// <summary>
        ///     Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.
        /// </summary>
        /// <value>
        ///     The strides.
        /// </value>
        [JsonIgnore]
        public Tuple<int, int> Strides
        {
            get => GetParam<Tuple<int, int>>("Strides");

            set => SetParam("Strides", value);
        }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return Convolution.MaxPool2D(inputFunction, PoolSize, Strides, Padding);
        }
    }
}