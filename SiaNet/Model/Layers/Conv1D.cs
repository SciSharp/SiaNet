using CNTK;
using Newtonsoft.Json;
using SiaNet.Model.Initializers;
using SiaNet.Model.Layers.Activations;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     1D convolution layer (e.g. temporal convolution). This layer creates a convolution kernel that is convolved with
    ///     the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If use_bias is True,
    ///     a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs
    ///     as well.
    /// </summary>
    /// <seealso cref="OptimizableLayerBase" />
    public class Conv1D : OptimizableLayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Conv1D" /> class.
        /// </summary>
        /// <param name="shape">The 1D input shape.</param>
        /// <param name="channels">Integer, the dimensionality of the output space</param>
        /// <param name="kernalSize">An integer specifying the length of the 1D convolution window.</param>
        /// <param name="strides">An integer specifying the stride length of the convolution.</param>
        /// <param name="padding">
        ///     Boolean, if true results in padding the input such that the output has the same length as the
        ///     original input.
        /// </param>
        /// <param name="dialation">
        ///     An integer specifying the dilation rate to use for dilated convolution. Currently, specifying
        ///     any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
        /// </param>
        /// <param name="activation">
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie.
        ///     "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations" />
        /// </param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">
        ///     Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers" />
        /// </param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers" /></param>
        public Conv1D(
            int channels,
            int kernalSize,
            int strides = 1,
            bool padding = true,
            int dialation = 1,
            ActivationBase activation = null,
            bool useBias = false,
            InitializerBase weightInitializer = null,
            InitializerBase biasInitializer = null)
        {
            WeightInitializer = weightInitializer ?? new Xavier();
            BiasInitializer = biasInitializer ?? new Zeros();
            Channels = channels;
            KernalSize = kernalSize;
            Padding = padding;
            Dialation = dialation;
            Activation = activation;
            UseBias = useBias;
            Strides = strides;
        }

        /// <summary>
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x)
        ///     = x). <see cref="SiaNet.Common.OptActivations" />
        /// </summary>
        /// <value>
        ///     The activation function name.
        /// </value>
        [JsonIgnore]
        public ActivationBase Activation
        {
            get => GetParam<ActivationBase>("Activation");

            set => SetParam("Activation", value);
        }

        /// <summary>
        ///     Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers" />
        /// </summary>
        /// <value>
        ///     The bias initializer.
        /// </value>
        [JsonIgnore]
        public InitializerBase BiasInitializer
        {
            get => GetParam<InitializerBase>("BiasInitializer");

            set => SetParam("BiasInitializer", value);
        }

        /// <summary>
        ///     Integer, the dimensionality of the output space
        /// </summary>
        /// <value>
        ///     The channels.
        /// </value>
        [JsonIgnore]
        public int Channels
        {
            get => GetParam<int>("Channels");

            set => SetParam("Channels", value);
        }

        /// <summary>
        ///     An integer specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate
        ///     value != 1 is incompatible with specifying any strides value != 1.
        /// </summary>
        /// <value>
        ///     The dialation.
        /// </value>
        [JsonIgnore]
        public int Dialation
        {
            get => GetParam<int>("Dialation");

            set => SetParam("Dialation", value);
        }

        /// <summary>
        ///     An integer specifying the length of the 1D convolution window.
        /// </summary>
        /// <value>
        ///     The size of the kernal.
        /// </value>
        [JsonIgnore]
        public int KernalSize
        {
            get => GetParam<int>("KernalSize");

            set => SetParam("KernalSize", value);
        }

        /// <summary>
        ///     Boolean, if true results in padding the input such that the output has the same length as the original input
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
        ///     An integer specifying the stride length of the convolution.
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

        /// <summary>
        ///     Boolean, whether the layer uses a bias vector.
        /// </summary>
        /// <value>
        ///     <c>true</c> if [use bias]; otherwise, <c>false</c>.
        /// </value>
        [JsonIgnore]
        public bool UseBias
        {
            get => GetParam<bool>("UseBias");

            set => SetParam("UseBias", value);
        }

        /// <summary>
        ///     Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers" />
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
            //if (inputFunction.Shape.Rank != 2)
            //{
            //    throw new ArgumentException("Variable has an invalid shape.", nameof(inputFunction));
            //}

            CNTK.Parameter convParams;
            CNTK.Function conv;

            if (inputFunction.Shape.Rank > 1)
            {
                var numInputChannels = inputFunction.Shape[inputFunction.Shape.Rank - 1];
                convParams = new CNTK.Parameter(new[] {KernalSize, numInputChannels, Channels}, DataType.Float,
                    WeightInitializer.ToDictionary(), GlobalParameters.Device);
                conv = CNTKLib.Convolution(convParams, inputFunction, new[] {Strides}, new BoolVector(new[] {true}),
                    new BoolVector(new[] {Padding, false, false}), new[] {Dialation});
            }
            else
            {
                convParams = new CNTK.Parameter(new[] {KernalSize, Channels}, DataType.Float,
                    WeightInitializer.ToDictionary(),
                    GlobalParameters.Device);
                conv = CNTKLib.Convolution(convParams, inputFunction, new[] {Strides}, new BoolVector(new[] {true}),
                    new BoolVector(new[] {Padding}), new[] {Dialation});
            }

            CNTK.Parameter bias;

            if (UseBias)
            {
                bias = new CNTK.Parameter(conv.Output.Shape, DataType.Float, BiasInitializer.ToDictionary(),
                    GlobalParameters.Device);
                conv = CNTKLib.Plus(bias, conv);
            }

            return Activation != null ? Activation.ToFunction((Function) conv) : (Function) conv;
        }
    }
}