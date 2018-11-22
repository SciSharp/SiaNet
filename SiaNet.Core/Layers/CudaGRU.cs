using System;
using System.Linq;
using CNTK;
using Newtonsoft.Json;
using SiaNet.Initializers;

namespace SiaNet.Layers
{
    /// <summary>
    ///     Gated recurrent units (GRUs) are a gating mechanism in recurrent neural networks,
    ///     introduced in 2014. Their performance on polyphonic music modeling and speech signal modeling was found to be
    ///     similar to that of long short-term memory. They have fewer parameters than LSTM, as they lack an output gate.
    /// </summary>
    /// <seealso cref="OptimizableLayerBase" />
    public class CudaGRU : OptimizableLayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="CudaGRU" /> class.
        /// </summary>
        /// <param name="hiddenSize">Size of the hidden layer.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="bidirectional">If bidirectional RNN</param>
        /// <param name="weightInitializer">The weight initializer.</param>
        public CudaGRU(
            uint hiddenSize,
            uint numLayers,
            bool bidirectional = false,
            InitializerBase weightInitializer = null)
        {
            LayerSize = hiddenSize;
            Layers = numLayers;
            BiDirectional = bidirectional;
            WeightInitializer = weightInitializer ?? new Xavier();
        }


        /// <summary>
        ///     Gets or sets a value indicating if the RNN is bi-directional.
        /// </summary>
        /// <value>
        ///     <c>true</c> if [bi-directional]; otherwise, <c>false</c>.
        /// </value>
        [JsonIgnore]
        public bool BiDirectional
        {
            get => GetParam<bool>("BiDirectional");

            set => SetParam("BiDirectional", value);
        }

        /// <summary>
        ///     Positive integer, number of layers
        /// </summary>
        /// <value>
        ///     The number of layers
        /// </value>
        [JsonIgnore]
        public uint Layers
        {
            get => GetParam<uint>("Layers");

            set => SetParam("Layers", value);
        }

        /// <summary>
        ///     Positive integer, the size of each layer
        /// </summary>
        /// <value>
        ///     The size of each layer
        /// </value>
        [JsonIgnore]
        public uint LayerSize
        {
            get => GetParam<uint>("LayerSize");

            set => SetParam("LayerSize", value);
        }


        /// <summary>
        ///     Initializer for the kernel weights matrix .
        /// </summary>
        /// <value>
        ///     The weight initializer.
        /// </value>
        [JsonIgnore]
        public InitializerBase WeightInitializer
        {
            get => GetParam<InitializerBase>("WeightInitializer");

            set => SetParam("WeightInitializer", value);
        }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            if (GlobalParameters.Device.Type != DeviceKind.GPU)
            {
                throw new NotSupportedException();
            }

            var s = inputFunction.Shape.Dimensions.ToArray();
            var weights = new CNTK.Parameter(s, DataType.Float, WeightInitializer.ToDictionary(),
                GlobalParameters.Device);

            return CNTKLib.OptimizedRNNStack(CNTK.Variable.InputVariable(s, DataType.Float), weights, LayerSize, Layers,
                BiDirectional, "gru");
        }
    }
}