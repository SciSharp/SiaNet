using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet.Common;
using SiaNet.Model.Initializers;

namespace SiaNet.Model.Layers
{
    /// <summary>
    /// Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Dense : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Dense"/> class.
        /// </summary>
        internal Dense()
        {
            base.Name = "Dense";
            base.Params = new ExpandoObject();
            Shape = null;
            Act = OptActivations.None;
            UseBias = false;
            WeightInitializer = new Xavier();
            BiasInitializer = new Zeros();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Dense"/> class.
        /// </summary>
        /// <param name="dim">Positive integer, dimensionality of the output space.</param>
        /// <param name="act">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        public Dense(int dim, string act = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
            : this()
        {
            Shape = null;
            Dim = dim;
            Act = act;
            UseBias = useBias;
            WeightInitializer = new BaseInitializer(weightInitializer);
            BiasInitializer = new BaseInitializer(biasInitializer);
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
        public Dense(int dim, int shape, string act = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
            : this(dim, act, useBias, weightInitializer, biasInitializer)
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
        public int? Shape
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

        /// <summary>
        /// Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x)
        /// </summary>
        /// <value>
        /// The activation function.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public string Act
        {
            get
            {
                return base.Params.Act;
            }

            set
            {
                base.Params.Act = value;
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
        public BaseInitializer WeightInitializer
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

        /// <summary>
        /// Initializer for the bias vector.
        /// </summary>
        /// <value>
        /// The bias initializer.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public BaseInitializer BiasInitializer
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
    }
}
