using Newtonsoft.Json;
using SiaNet.Common;
using SiaNet.Model.Initializers;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise
    ///     activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is
    ///     a bias vector created by the layer (only applicable if use_bias is True).
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Dense : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Dense" /> class.
        /// </summary>
        /// <param name="dim">Positive integer, dimensionality of the output space..</param>
        /// <param name="shape">The input shape.</param>
        /// <param name="act">
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie. "linear"
        ///     activation: a(x) = x). <see cref="SiaNet.Common.OptActivations" />
        /// </param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. </param>
        /// <param name="biasInitializer">Initializer for the bias vector. </param>
        public Dense(
            int dim,
            int? shape = null,
            string act = OptActivations.None,
            bool useBias = false,
            object weightInitializer = null,
            object biasInitializer = null)
            : this()
        {
            Shape = shape;
            Dim = dim;
            Act = act;
            UseBias = useBias;
            WeightInitializer = Utility.GetInitializerFromObject(weightInitializer, new Xavier());
            BiasInitializer = Utility.GetInitializerFromObject(biasInitializer, new Zeros());
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Dense" /> class.
        /// </summary>
        internal Dense()
        {
            Name = "Dense";
        }

        /// <summary>
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x)
        ///     = x)
        /// </summary>
        /// <value>
        ///     The activation function.
        /// </value>
        [JsonIgnore]
        public string Act
        {
            get => GetParam<string>("Act");

            set => SetParam("Act", value);
        }

        /// <summary>
        ///     Initializer for the bias vector.
        /// </summary>
        /// <value>
        ///     The bias initializer.
        /// </value>
        [JsonIgnore]
        public Initializer BiasInitializer
        {
            get => GetParam<Initializer>("BiasInitializer");

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
        ///     The input shape for this layer
        /// </summary>
        /// <value>
        ///     The shape.
        /// </value>
        [JsonIgnore]
        public int? Shape
        {
            get => GetParam<int?>("Shape");

            set => SetParam("Shape", value);
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
        public Initializer WeightInitializer
        {
            get => GetParam<Initializer>("WeightInitializer");

            set => SetParam("WeightInitializer", value);
        }
    }
}