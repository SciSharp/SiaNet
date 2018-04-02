using System;
using CNTK;
using Newtonsoft.Json;
using SiaNet.Common;
using SiaNet.Model.Initializers;
using SiaNet.NN;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Long short-term memory (LSTM) is a recurrent neural network (RNN) architecture that remembers values over arbitrary
    ///     intervals
    /// </summary>
    /// <seealso cref="OptimizableLayerBase" />
    public class LSTM : OptimizableLayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="LSTM" /> class.
        /// </summary>
        /// <param name="dim">Positive integer, dimensionality of the output space.</param>
        /// <param name="shape">The input shape.</param>
        /// <param name="activation">
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie.
        ///     "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations" />
        /// </param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">
        ///     Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers" />
        /// </param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers" /></param>
        public LSTM(
            int dim,
            int? cellDim = null,
            string activation = OptActivations.Tanh,
            string recurrentActivation = OptActivations.Sigmoid,
            object weightInitializer = null,
            object recurrentInitializer = null,
            bool useBias = true,
            object biasInitializer = null,
            bool returnSequence = false)
        {
            Dim = dim;
            CellDim = cellDim;
            Activation = activation;
            RecurrentActivation = recurrentActivation;
            UseBias = useBias;
            ReturnSequence = returnSequence;
            WeightInitializer = Utility.GetInitializerFromObject(weightInitializer, new GlorotUniform());
            RecurrentInitializer = Utility.GetInitializerFromObject(recurrentInitializer, new GlorotUniform());
            BiasInitializer = Utility.GetInitializerFromObject(biasInitializer, new Zeros());
        }


        /// <summary>
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x)
        ///     = x)
        /// </summary>
        /// <value>
        ///     The activation function.
        /// </value>
        [JsonIgnore]
        public string Activation
        {
            get => GetParam<string>("Activation");

            set => SetParam("Activation", value);
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

        public int? CellDim
        {
            get => GetParam<int?>("CellDim");

            set => SetParam("CellDim", value);
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

        [JsonIgnore]
        public string RecurrentActivation
        {
            get => GetParam<string>("RecurrentActivation");

            set => SetParam("RecurrentActivation", value);
        }

        public Initializer RecurrentInitializer
        {
            get => GetParam<Initializer>("RecurrentInitializer");

            set => SetParam("RecurrentInitializer", value);
        }

        /// <summary>
        ///     Gets or sets a value indicating whether [return sequence].
        /// </summary>
        /// <value>
        ///     <c>true</c> if [return sequence]; otherwise, <c>false</c>.
        /// </value>
        public bool ReturnSequence
        {
            get => GetParam<bool>("ReturnSequence");

            set => SetParam("ReturnSequence", value);
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

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            //if (inputFunction.Shape.Rank < 3)
            //{
            //    throw new ArgumentException("Variable has an invalid shape.", nameof(inputFunction));
            //}

            return Recurrent.LSTM(inputFunction, Dim, CellDim, Activation, RecurrentActivation, WeightInitializer,
                RecurrentInitializer, UseBias, BiasInitializer, ReturnSequence);
        }
    }
}