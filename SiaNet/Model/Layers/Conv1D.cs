using SiaNet.Common;
using SiaNet.Model.Initializers;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    /// <summary>
    /// 1D convolution layer (e.g. temporal convolution). This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Conv1D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Conv1D"/> class.
        /// </summary>
        internal Conv1D()
        {
            base.Name = "Conv1D";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Conv1D"/> class.
        /// </summary>
        /// <param name="shape">The 1D input shape.</param>
        /// <param name="channels">Integer, the dimensionality of the output space</param>
        /// <param name="kernalSize">An integer specifying the length of the 1D convolution window.</param>
        /// <param name="strides">An integer specifying the stride length of the convolution.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <param name="dialation">An integer specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        public Conv1D(Tuple<int, int> shape, int channels, int kernalSize, int strides = 1, bool padding = true, int dialation = 1, string activation = OptActivations.None, bool useBias = false, object weightInitializer = null, object biasInitializer = null)
            : this()
        {
            WeightInitializer = Utility.GetInitializerFromObject(weightInitializer, new Xavier());
            BiasInitializer = Utility.GetInitializerFromObject(biasInitializer, new Zeros());
            Shape = Tuple.Create<int, int>(shape.Item1, shape.Item2);
            Channels = channels;
            KernalSize = kernalSize;
            Padding = padding;
            Dialation = dialation;
            Act = activation;
            UseBias = useBias;
            Strides = strides;
        }

        /// <summary>
        /// The 1D input shape
        /// </summary>
        /// <value>
        /// The shape.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int> Shape
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
        /// Integer, the dimensionality of the output space
        /// </summary>
        /// <value>
        /// The channels.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int Channels
        {
            get
            {
                return base.Params.Channels;
            }

            set
            {
                base.Params.Channels = value;
            }
        }

        /// <summary>
        /// An integer specifying the length of the 1D convolution window.
        /// </summary>
        /// <value>
        /// The size of the kernal.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int KernalSize
        {
            get
            {
                return base.Params.KernalSize;
            }

            set
            {
                base.Params.KernalSize = value;
            }
        }

        /// <summary>
        /// An integer specifying the stride length of the convolution.
        /// </summary>
        /// <value>
        /// The strides.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int Strides
        {
            get
            {
                return base.Params.Strides;
            }

            set
            {
                base.Params.Strides = value;
            }
        }

        /// <summary>
        /// Boolean, if true results in padding the input such that the output has the same length as the original input
        /// </summary>
        /// <value>
        ///   <c>true</c> if padding; otherwise, <c>false</c>.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public bool Padding
        {
            get
            {
                return base.Params.Padding;
            }

            set
            {
                base.Params.Padding = value;
            }
        }

        /// <summary>
        /// An integer specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
        /// </summary>
        /// <value>
        /// The dialation.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int Dialation
        {
            get
            {
                return base.Params.Dialation;
            }

            set
            {
                base.Params.Dialation = value;
            }
        }

        /// <summary>
        /// Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/>
        /// </summary>
        /// <value>
        /// The activation function name.
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
        /// Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/>
        /// </summary>
        /// <value>
        /// The weight initializer.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Initializer WeightInitializer
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
        /// Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/>
        /// </summary>
        /// <value>
        /// The bias initializer.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Initializer BiasInitializer
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
