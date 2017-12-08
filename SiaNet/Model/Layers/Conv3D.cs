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
    /// 3D convolution layer (e.g. spatial convolution over volumes). This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If  use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Conv3D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Conv3D"/> class.
        /// </summary>
        internal Conv3D()
        {
            base.Name = "Conv3D";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Conv3D"/> class.
        /// </summary>
        /// <param name="shape">The 3D input shape.</param>
        /// <param name="channels">Integer, the dimensionality of the output space.</param>
        /// <param name="kernalSize">A tuple of 3 integers, specifying the depth, height and width of the 3D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides">A tuple of 3 integers, specifying the strides of the convolution along each spatial dimension. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <param name="dialation">A tuple of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        public Conv3D(int channels, Tuple<int, int, int> kernalSize, Tuple<int, int, int, int> shape = null, Tuple<int, int, int> strides = null, bool padding = true, Tuple<int, int, int> dialation = null,
            string activation = OptActivations.None, bool useBias = false, object weightInitializer = null, object biasInitializer = null)
            : this()
        {
            Shape = shape;
            Channels = channels;
            KernalSize = kernalSize;
            Strides = strides == null ? Tuple.Create(1, 1, 1) : strides;
            Padding = padding;
            Dialation = dialation == null ? Tuple.Create(1, 1, 1) : dialation;
            Act = activation;
            UseBias = useBias;
            WeightInitializer = Utility.GetInitializerFromObject(weightInitializer, new Xavier());
            BiasInitializer = Utility.GetInitializerFromObject(biasInitializer, new Zeros());
        }

        /// <summary>
        /// The 3D input shape.
        /// </summary>
        /// <value>
        /// The shape.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int, int, int> Shape
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
        /// Integer, the dimensionality of the output space.
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
        /// A tuple of 3 integers, specifying the depth, height and width of the 3D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
        /// </summary>
        /// <value>
        /// The size of the kernal.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int, int> KernalSize
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
        /// A tuple of 3 integers, specifying the strides of the convolution along each spatial dimension. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
        /// </summary>
        /// <value>
        /// The strides.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int, int> Strides
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
        /// Boolean, if true results in padding the input such that the output has the same length as the original input.
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
        /// A tuple of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
        /// </summary>
        /// <value>
        /// The dialation.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int, int> Dialation
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
