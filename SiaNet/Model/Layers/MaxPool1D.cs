using CNTK;
using Newtonsoft.Json;
using SiaNet.NN;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Max pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class MaxPool1D : LayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="MaxPool1D" /> class.
        /// </summary>
        /// <param name="poolSize">Integer, size of the max pooling windows.</param>
        /// <param name="strides">Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.</param>
        /// <param name="padding">
        ///     Boolean, if true results in padding the input such that the output has the same length as the
        ///     original input.
        /// </param>
        public MaxPool1D(int poolSize, int strides = 1, bool padding = true)
            : this()
        {
            PoolSize = poolSize;
            Strides = strides;
            Padding = padding;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="MaxPool1D" /> class.
        /// </summary>
        internal MaxPool1D()
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
        ///     Integer, size of the max pooling windows.
        /// </summary>
        /// <value>
        ///     The size of the pool.
        /// </value>
        [JsonIgnore]
        public int PoolSize
        {
            get => GetParam<int>("PoolSize");

            set => SetParam("PoolSize", value);
        }

        /// <summary>
        ///     Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.
        /// </summary>
        /// <value>
        ///     The strides.
        /// </value>
        [JsonIgnore]
        public int Strides
        {
            get => GetParam<int>("Strides");

            set => SetParam("Strides", value);
        }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return Convolution.MaxPool1D(inputFunction, PoolSize, Strides, Padding);
        }
    }
}