using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet.Common;

namespace SiaNet.Model.Layers
{
    /// <summary>
    /// Long short-term memory (LSTM) is a recurrent neural network (RNN) architecture that remembers values over arbitrary intervals
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class LSTM : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LSTM"/> class.
        /// </summary>
        internal LSTM()
        {
            base.Name = "LSTM";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Dense"/> class.
        /// </summary>
        /// <param name="dim">Positive integer, dimensionality of the output space.</param>
        /// <param name="act">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        public LSTM(int dim, int? cellDim = null, string activation = OptActivations.Tanh, string recurrentActivation = OptActivations.Sigmoid, string weightInitializer = OptInitializers.GlorotUniform, string recurrentInitializer = OptInitializers.GlorotUniform, bool useBias = true, string biasInitializer = OptInitializers.Zeros, bool returnSequence = false)
            : this()
        {
            Shape = null;
            Dim = dim;
            CellDim = cellDim;
            Activation = activation;
            RecurrentActivation = recurrentActivation;
            WeightInitializer = weightInitializer;
            RecurrentInitializer = recurrentInitializer;
            UseBias = useBias;
            BiasInitializer = biasInitializer;
            ReturnSequence = returnSequence;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Dense"/> class.
        /// </summary>
        /// <param name="dim">Positive integer, dimensionality of the output space.</param>
        /// <param name="act">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        public LSTM(int dim, string activation = OptActivations.Tanh, string recurrentActivation = OptActivations.Sigmoid, string weightInitializer = OptInitializers.GlorotUniform, string recurrentInitializer = OptInitializers.GlorotUniform, bool useBias = true, string biasInitializer = OptInitializers.Zeros, bool returnSequence = false)
            : this()
        {
            Shape = null;
            Dim = dim;
            CellDim = null;
            Activation = activation;
            RecurrentActivation = recurrentActivation;
            WeightInitializer = weightInitializer;
            RecurrentInitializer = recurrentInitializer;
            UseBias = useBias;
            BiasInitializer = biasInitializer;
            ReturnSequence = returnSequence;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Dense"/> class.
        /// </summary>
        /// <param name="dim">Positive integer, dimensionality of the output space..</param>
        /// <param name="shape">The input shape.</param>
        /// <param name="act">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        public LSTM(int dim, int[] shape = null, int? cellDim = null, string activation = OptActivations.Tanh, string recurrentActivation = OptActivations.Sigmoid, string weightInitializer = OptInitializers.GlorotUniform, string recurrentInitializer = OptInitializers.GlorotUniform, bool useBias = true, string biasInitializer = OptInitializers.Zeros, bool returnSequence = false)
            : this(dim, cellDim, activation, recurrentActivation, weightInitializer, recurrentInitializer, useBias, biasInitializer, returnSequence)
        {
            Shape = shape;
        }

        /// <summary>
        /// The input shape for this layer
        /// </summary>
        /// <value>
        /// The shape.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int[] Shape
        {
            get
            {
                return base.Params.Shape;
            }

            set
            {
                base.Params.Shape = value;
            }
        }

        /// <summary>
        /// Positive integer, dimensionality of the output space.
        /// </summary>
        /// <value>
        /// The output dimension.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int Dim
        {
            get
            {
                return base.Params.Dim;
            }

            set
            {
                base.Params.Dim = value;
            }
        }

        public int? CellDim
        {
            get
            {
                return base.Params.CellDim;
            }

            set
            {
                base.Params.CellDim = value;
            }
        }

        /// <summary>
        /// Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x)
        /// </summary>
        /// <value>
        /// The activation function.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public string Activation
        {
            get
            {
                return base.Params.Activation;
            }

            set
            {
                base.Params.Activation = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public string RecurrentActivation
        {
            get
            {
                return base.Params.RecurrentActivation;
            }

            set
            {
                base.Params.RecurrentActivation = value;
            }
        }

        /// <summary>
        /// Boolean, whether the layer uses a bias vector.
        /// </summary>
        /// <value>
        ///   <c>true</c> if [use bias]; otherwise, <c>false</c>.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public bool UseBias
        {
            get
            {
                return base.Params.UseBias;
            }

            set
            {
                base.Params.UseBias = value;
            }
        }

        /// <summary>
        /// Initializer for the kernel weights matrix .
        /// </summary>
        /// <value>
        /// The weight initializer.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public string WeightInitializer
        {
            get
            {
                return base.Params.WeightInitializer;
            }

            set
            {
                base.Params.WeightInitializer = value;
            }
        }

        public string RecurrentInitializer
        {
            get
            {
                return base.Params.RecurrentInitializer;
            }

            set
            {
                base.Params.RecurrentInitializer = value;
            }
        }

        /// <summary>
        /// Initializer for the bias vector.
        /// </summary>
        /// <value>
        /// The bias initializer.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public string BiasInitializer
        {
            get
            {
                return base.Params.BiasInitializer;
            }

            set
            {
                base.Params.BiasInitializer = value;
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether [return sequence].
        /// </summary>
        /// <value>
        ///   <c>true</c> if [return sequence]; otherwise, <c>false</c>.
        /// </value>
        public bool ReturnSequence
        {
            get
            {
                return base.Params.ReturnSequence;
            }

            set
            {
                base.Params.ReturnSequence = value;
            }
        }
    }
}
