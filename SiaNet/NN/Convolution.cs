namespace SiaNet.NN
{
    using CNTK;
    using System;
    using SiaNet.Common;

    /// <summary>
    /// Convolutional Neural Networks are very similar to ordinary Neural Networks. They are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.
    /// </summary>
    public class Convolution
    {
        /// <summary>
        /// 1D convolution layer (e.g. temporal convolution). This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="channels">Integer, the dimensionality of the output space</param>
        /// <param name="kernalSize">An integer specifying the length of the 1D convolution window.</param>
        /// <param name="strides">An integer specifying the stride length of the convolution.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <param name="dialation">An integer specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <returns></returns>
        public static Function Conv1D(Variable layer, int channels, int kernalSize, int strides=1, bool padding=true, int dialation=1, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            int numInputChannels = layer.Shape[layer.Shape.Rank - 1];
            var convParams = new Parameter(new int[] { kernalSize, numInputChannels, channels }, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);
            var conv = CNTKLib.Convolution(convParams, layer, new int[] { strides }, new BoolVector(new bool[] { true, false, false }), new BoolVector(new bool[] { padding, false, false }), new int[] { dialation });

            Parameter bias = null;
            if (useBias)
            {
                bias = new Parameter(conv.Output.Shape, DataType.Float, Initializers.Get(biasInitializer), GlobalParameters.Device);
                conv = CNTKLib.Plus(bias, conv);
            }
            
            return Basic.Activation(conv, activation);
        }

        /// <summary>
        /// 1D convolution layer (e.g. temporal convolution). This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
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
        /// <returns></returns>
        public static Function Conv1D(Tuple<int, int> shape, int channels, int kernalSize, int strides = 1, bool padding = true, int dialation = 1, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            Variable input = CNTKLib.InputVariable(new int[] { shape.Item1, shape.Item2 }, DataType.Float);
            return Conv1D(input, channels, kernalSize, strides, padding, dialation, activation, useBias, weightInitializer, biasInitializer);
        }

        /// <summary>
        /// 2D convolution layer (e.g. spatial convolution over images). This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If  use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="channels">Integer, the dimensionality of the output space.</param>
        /// <param name="kernalSize">A tuple of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides">A tuple of 2 integers, specifying the strides of the convolution along the width and height. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <param name="dialation">A tuple of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <returns></returns>
        public static Function Conv2D(Variable layer, int channels, Tuple<int, int> kernalSize, Tuple<int, int> strides, bool padding = true, Tuple<int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            int numInputChannels = layer.Shape[layer.Shape.Rank - 1];
            var convParams = new Parameter(new int[] { kernalSize.Item1, kernalSize.Item2, numInputChannels, channels }, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);
            if (dialation == null)
            {
                dialation = new Tuple<int, int>(1, 1);
            }

            var conv = CNTKLib.Convolution(convParams, layer, new int[] { strides.Item1, strides.Item2 }, new BoolVector(new bool[] { true }), new BoolVector(new bool[] { padding, padding, false }), new int[] { dialation.Item1, dialation.Item2 });

            Parameter bias = null;
            if (useBias)
            {
                bias = new Parameter(conv.Output.Shape, DataType.Float, Initializers.Get(biasInitializer), GlobalParameters.Device);
                conv = CNTKLib.Plus(bias, conv);
            }

            return Basic.Activation(conv, activation);
        }

        /// <summary>
        /// Conv2s the d.2D convolution layer (e.g. spatial convolution over images). This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If  use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
        /// </summary>
        /// <param name="shape">The 2D input shape.</param>
        /// <param name="channels">Integer, the dimensionality of the output space.</param>
        /// <param name="kernalSize">A tuple of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides">A tuple of 2 integers, specifying the strides of the convolution along the width and height. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <param name="dialation">A tuple of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <returns></returns>
        public static Function Conv2D(Tuple<int, int, int> shape, int channels, Tuple<int, int> kernalSize, Tuple<int, int> strides, bool padding = true, Tuple<int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            Variable input = CNTKLib.InputVariable(new int[] { shape.Item1, shape.Item2, shape.Item3 }, DataType.Float);
            return Conv2D(input, channels, kernalSize, strides, padding, dialation, activation, useBias, weightInitializer, biasInitializer);
        }

        /// <summary>
        /// 3D convolution layer (e.g. spatial convolution over volumes). This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If  use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="channels">Integer, the dimensionality of the output space.</param>
        /// <param name="kernalSize">A tuple of 3 integers, specifying the depth, height and width of the 3D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides">A tuple of 3 integers, specifying the strides of the convolution along each spatial dimension. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <param name="dialation">A tuple of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <returns></returns>
        public static Function Conv3D(Variable layer, int channels, Tuple<int, int, int> kernalSize, Tuple<int, int, int> strides, bool padding = true, Tuple<int, int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            int numInputChannels = layer.Shape[layer.Shape.Rank - 1];
            var convParams = new Parameter(new int[] { kernalSize.Item1, kernalSize.Item2, kernalSize.Item3, numInputChannels, channels }, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);

            if(dialation==null)
            {
                dialation = new Tuple<int, int, int>(1, 1, 1);
            }

            var conv = CNTKLib.Convolution(convParams, layer, new int[] { strides.Item1, strides.Item2, strides.Item3 }, new BoolVector(new bool[] { true, true, true }), new BoolVector(new bool[] { padding, padding, padding }), new int[] { dialation.Item1, dialation.Item2, dialation.Item3 });
            Parameter bias = null;
            if (useBias)
            {
                bias = new Parameter(conv.Output.Shape, DataType.Float, Initializers.Get(biasInitializer), GlobalParameters.Device);
                conv = CNTKLib.Plus(bias, conv);
            }

            return Basic.Activation(conv, activation);
        }

        /// <summary>
        /// 3D convolution layer (e.g. spatial convolution over volumes). This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If  use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
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
        /// <returns></returns>
        public static Function Conv3D(Tuple<int, int, int, int> shape, int channels, Tuple<int, int, int> kernalSize, Tuple<int, int, int> strides, bool padding = true, Tuple<int, int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            Variable input = CNTKLib.InputVariable(new int[] { shape.Item1, shape.Item2, shape.Item3 }, DataType.Float);
            return Conv3D(input, channels, kernalSize, strides, padding, dialation, activation, useBias, weightInitializer, biasInitializer);
        }

        /// <summary>
        /// Max pooling operation for temporal data.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="poolSize">Integer, size of the max pooling windows.</param>
        /// <param name="strides">Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <returns></returns>
        public static Function MaxPool1D(Variable layer, int poolSize, int strides, bool padding=true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { poolSize }, new int[] { strides }, new BoolVector(new bool[] { padding, false, false }));
        }

        /// <summary>
        /// Max pooling operation for spatial data.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="poolSize">A tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.</param>
        /// <param name="strides">Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <returns></returns>
        public static Function MaxPool2D(Variable layer, Tuple<int, int> poolSize, Tuple<int, int> strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { poolSize.Item1, poolSize.Item2 }, new int[] { strides.Item1, strides.Item2 }, new BoolVector(new bool[] { padding, padding, false }));
        }

        /// <summary>
        /// Max pooling operation for 3D data (spatial or spatio-temporal).
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="poolSize">Tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each dimension.</param>
        /// <param name="strides">Tuple of 3 integers, or None. Strides values.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <returns></returns>
        public static Function MaxPool3D(Variable layer, Tuple<int, int, int> poolSize, Tuple<int, int, int> strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { poolSize.Item1, poolSize.Item2, poolSize.Item3 }, new int[] { strides.Item1, strides.Item2, strides.Item3 }, new BoolVector(new bool[] { padding, padding, padding }));
        }

        /// <summary>
        /// Average pooling operation for spatial data.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="poolSize">A tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.</param>
        /// <param name="strides">Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <returns></returns>
        public static Function AvgPool1D(Variable layer, int poolSize, int strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { poolSize }, new int[] { strides }, new BoolVector(new bool[] { padding, false, false }));
        }

        /// <summary>
        /// Average pooling operation for spatial data.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="poolSize">A tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.</param>
        /// <param name="strides">Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <returns></returns>
        public static Function AvgPool2D(Variable layer, Tuple<int, int> poolSize, Tuple<int, int> strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { poolSize.Item1, poolSize.Item2 }, new int[] { strides.Item1, strides.Item2 }, new BoolVector(new bool[] { padding, padding, false }));
        }

        /// <summary>
        /// Average pooling operation for 3D data (spatial or spatio-temporal).
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="poolSize">Tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each dimension.</param>
        /// <param name="strides">Tuple of 3 integers, or None. Strides values.</param>
        /// <param name="padding">Boolean, if true results in padding the input such that the output has the same length as the original input.</param>
        /// <returns></returns>
        public static Function AvgPool3D(Variable layer, Tuple<int, int, int> poolSize, Tuple<int, int, int> strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { poolSize.Item1, poolSize.Item2, poolSize.Item3 }, new int[] { strides.Item1, strides.Item2, strides.Item3 }, new BoolVector(new bool[] { padding, padding, padding }));
        }

        /// <summary>
        /// Global max pooling operation for temporal data.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <returns></returns>
        public static Function GlobalMaxPool1D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { layer.Shape[0] });
        }

        /// <summary>
        /// Global max pooling operation for spatial data.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <returns></returns>
        public static Function GlobalMaxPool2D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { layer.Shape[0], layer.Shape[1] });
        }

        /// <summary>
        /// Global max pooling 3D data (spatial or spatio-temporal).
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <returns></returns>
        public static Function GlobalMaxPool3D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { layer.Shape[0], layer.Shape[1], layer.Shape[2] });
        }

        /// <summary>
        /// Global average pooling operation for temporal data.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <returns></returns>
        public static Function GlobalAvgPool1D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { layer.Shape[0] }); 
        }

        /// <summary>
        /// Global average pooling operation for spatial data.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <returns></returns>
        public static Function GlobalAvgPool2D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { layer.Shape[0], layer.Shape[1] });
        }

        /// <summary>
        /// Global average pooling 3D data (spatial or spatio-temporal).
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <returns></returns>
        public static Function GlobalAvgPool3D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { layer.Shape[0], layer.Shape[1], layer.Shape[2] });
        }
    }
}
