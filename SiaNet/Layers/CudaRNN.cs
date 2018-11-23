using System;
using System.Linq;
using CNTK;
using Newtonsoft.Json;
using SiaNet.Initializers;
using SiaNet.Layers.Activations;

namespace SiaNet.Layers
{
    /// <summary>
    ///     A recurrent neural network (RNN) is a class of artificial neural network where connections between units form a
    ///     directed cycle. This allows it to exhibit dynamic temporal behavior. Unlike feed-forward neural networks, RNNs can
    ///     use their internal memory to process arbitrary sequences of inputs.
    /// </summary>
    /// <seealso cref="OptimizableLayerBase" />
    public class CudaRNN : OptimizableLayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="CudaRNN" /> class.
        /// </summary>
        /// <param name="hiddenSize">Size of the hidden layer.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="bidirectional">If bidirectional RNN</param>
        /// <param name="activation">Activation function to use. Supported are ReLU and TanH</param>
        /// <param name="weightInitializer">The weight initializer.</param>
        public CudaRNN(
            uint hiddenSize,
            uint numLayers,
            bool bidirectional = false,
            ActivationBase activation = null,
            InitializerBase weightInitializer = null)
        {
            LayerSize = hiddenSize;
            Layers = numLayers;
            BiDirectional = bidirectional;
            WeightInitializer = weightInitializer ?? new Xavier();
            Activation = activation ?? new ReLU();

            if (!(Activation is ReLU) && !(Activation is Tanh))
            {
                throw new NotSupportedException("Supported activation for RNN is ReLU and Tanh");
            }
        }

        /// <summary>
        ///     Gets or sets the activation function to use.
        /// </summary>
        [JsonIgnore]
        public ActivationBase Activation
        {
            get => GetParam<ActivationBase>("Activation");

            set => SetParam("Activation", value);
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

            string rnnName;

            if (Activation is ReLU)
            {
                rnnName = "rnnReLU";
            }
            else if (Activation is Tanh)
            {
                rnnName = "rnnTanh";
            }
            else
            {
                throw new NotSupportedException("Supported activation for RNN is ReLU and Tanh");
            }

            var s = inputFunction.Shape.Dimensions.ToArray();
            var weights = new CNTK.Parameter(s, DataType.Float, WeightInitializer.ToDictionary(),
                GlobalParameters.Device);

            return CNTKLib.OptimizedRNNStack(CNTK.Variable.InputVariable(s, DataType.Float), weights, LayerSize, Layers,
                BiDirectional, rnnName);
        }
    }
}