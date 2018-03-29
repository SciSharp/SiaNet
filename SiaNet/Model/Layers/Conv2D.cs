using System;
using Newtonsoft.Json;
using SiaNet.Common;
using SiaNet.Model.Initializers;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     2D convolution layer (e.g. spatial convolution over images). This layer creates a convolution kernel that is
    ///     convolved with the layer input to produce a tensor of outputs. If  use_bias is True, a bias vector is created and
    ///     added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Conv2D : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Conv2D" /> class.
        /// </summary>
        /// <param name="shape">The 2D input shape.</param>
        /// <param name="channels">Integer, the dimensionality of the output space.</param>
        /// <param name="kernalSize">
        ///     A tuple of 2 integers, specifying the width and height of the 2D convolution window. Can be a
        ///     single integer to specify the same value for all spatial dimensions.
        /// </param>
        /// <param name="strides">
        ///     A tuple of 2 integers, specifying the strides of the convolution along the width and height. Can
        ///     be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is
        ///     incompatible with specifying any dilation_rate value != 1.
        /// </param>
        /// <param name="padding">
        ///     Boolean, if true results in padding the input such that the output has the same length as the
        ///     original input.
        /// </param>
        /// <param name="dialation">
        ///     A tuple of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a
        ///     single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value
        ///     != 1 is incompatible with specifying any stride value != 1.
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
        public Conv2D(
            int channels,
            Tuple<int, int> kernalSize,
            Tuple<int, int, int> shape = null,
            Tuple<int, int> strides = null,
            bool padding = true,
            Tuple<int, int> dialation = null,
            string activation = OptActivations.None,
            bool useBias = false,
            object weightInitializer = null,
            object biasInitializer = null)
            : this()
        {
            Shape = shape;
            Channels = channels;
            KernalSize = kernalSize;
            Strides = strides == null ? Tuple.Create(1, 1) : strides;
            Padding = padding;
            Dialation = dialation == null ? Tuple.Create(1, 1) : dialation;
            Act = activation;
            UseBias = useBias;
            WeightInitializer = Utility.GetInitializerFromObject(weightInitializer, new Xavier());
            BiasInitializer = Utility.GetInitializerFromObject(biasInitializer, new Zeros());
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Conv2D" /> class.
        /// </summary>
        internal Conv2D()
        {
            Name = "Conv2D";
        }

        /// <summary>
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x)
        ///     = x). <see cref="SiaNet.Common.OptActivations" />
        /// </summary>
        /// <value>
        ///     The activation function name.
        /// </value>
        [JsonIgnore]
        public string Act
        {
            get => GetParam<string>("Act");

            set => SetParam("Act", value);
        }

        /// <summary>
        ///     Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers" />
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
        ///     Gets or sets the channels.Integer, the dimensionality of the output space.
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
        ///     A tuple of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to
        ///     specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is
        ///     incompatible with specifying any stride value != 1.
        /// </summary>
        /// <value>
        ///     The dialation.
        /// </value>
        [JsonIgnore]
        public Tuple<int, int> Dialation
        {
            get => GetParam<Tuple<int, int>>("Dialation");

            set => SetParam("Dialation", value);
        }

        /// <summary>
        ///     A tuple of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to
        ///     specify the same value for all spatial dimensions.
        /// </summary>
        /// <value>
        ///     The size of the kernal.
        /// </value>
        [JsonIgnore]
        public Tuple<int, int> KernalSize
        {
            get => GetParam<Tuple<int, int>>("KernalSize");

            set => SetParam("KernalSize", value);
        }

        /// <summary>
        ///     Boolean, if true results in padding the input such that the output has the same length as the original input.
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
        ///     The 2D input shape
        /// </summary>
        /// <value>
        ///     The shape.
        /// </value>
        [JsonIgnore]
        public Tuple<int, int, int> Shape
        {
            get => GetParam<Tuple<int, int, int>>("Shape");

            set => SetParam("Shape", value);
        }

        /// <summary>
        ///     A tuple of 2 integers, specifying the strides of the convolution along the width and height. Can be a single
        ///     integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with
        ///     specifying any dilation_rate value != 1.
        /// </summary>
        /// <value>
        ///     The strides.
        /// </value>
        [JsonIgnore]
        public Tuple<int, int> Strides
        {
            get => GetParam<Tuple<int, int>>("Strides");

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
        public Initializer WeightInitializer
        {
            get => GetParam<Initializer>("WeightInitializer");

            set => SetParam("WeightInitializer", value);
        }
    }
}