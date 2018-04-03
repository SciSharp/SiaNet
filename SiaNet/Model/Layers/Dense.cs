using System.Linq;
using CNTK;
using Newtonsoft.Json;
using SiaNet.Model.Initializers;
using SiaNet.Model.Layers.Activations;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise
    ///     activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is
    ///     a bias vector created by the layer (only applicable if use_bias is True).
    /// </summary>
    /// <seealso cref="OptimizableLayerBase" />
    public class Dense : OptimizableLayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Dense" /> class.
        /// </summary>
        /// <param name="dim">Positive integer, dimensionality of the output space..</param>
        /// <param name="activation">
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie. "linear"
        ///     activation: a(x) = x). <see cref="SiaNet.Common.OptActivations" />
        /// </param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. </param>
        /// <param name="biasInitializer">Initializer for the bias vector. </param>
        public Dense(
            int dim,
            ActivationBase activation = null,
            bool useBias = false,
            InitializerBase weightInitializer = null,
            InitializerBase biasInitializer = null)
        {
            Dim = dim;
            Activation = activation;
            UseBias = useBias;
            WeightInitializer = weightInitializer ?? new Xavier();
            BiasInitializer = biasInitializer ?? new Zeros();
        }


        /// <summary>
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x)
        ///     = x)
        /// </summary>
        /// <value>
        ///     The activation function.
        /// </value>
        [JsonIgnore]
        public ActivationBase Activation
        {
            get => GetParam<ActivationBase>("Activation");

            set => SetParam("Activation", value);
        }

        /// <summary>
        ///     Initializer for the bias vector.
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
        ///     Positive integer, dimensionality of the output space.
        /// </summary>
        /// <value>
        ///     The output dimension.
        /// </value>
        [JsonIgnore]
        public int Dim
        {
            get => GetParam<int>("Dim");

            set => SetParam("Dim", value);
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
            //if (inputFunction.Shape.Rank != 1)
            //{
            //    throw new ArgumentException("Variable has an invalid shape.", nameof(inputFunction));
            //}

            if (inputFunction.Shape.Rank != 1)
            {
                var newDim = inputFunction.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                inputFunction = (Function) CNTKLib.Reshape(inputFunction, new[] {newDim});
            }

            var inputDim = inputFunction.Shape[0];

            int[] s = {Dim, inputDim};
            var weights = new CNTK.Parameter(s, DataType.Float, WeightInitializer.ToDictionary(),
                GlobalParameters.Device);

            var bias = UseBias
                ? new CNTK.Parameter(new Shape(Dim), DataType.Float, BiasInitializer.ToDictionary(),
                    GlobalParameters.Device)
                : new CNTK.Parameter(new Shape(Dim), DataType.Float, 0.0f, GlobalParameters.Device);

            var fullyConnected = CNTKLib.Plus(bias, CNTKLib.Times(weights, inputFunction));

            return Activation != null ? Activation.ToFunction((Function) fullyConnected) : (Function) fullyConnected;
        }
    }
}