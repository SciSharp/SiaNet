namespace SiaNet.NN
{
    using CNTK;
    using SiaNet.Common;
    using System.Linq;

    /// <summary>
    /// Functions to implement the basic layers like Dense, Activation, Dropout etc. They are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity.
    /// </summary>
    public class Basic
    {
        /// <summary>
        /// Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
        /// </summary>
        /// <param name="shape">The input shape.</param>
        /// <param name="dim">Positive integer, dimensionality of the output space..</param>
        /// <param name="act">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <returns></returns>
        public static Function Dense(int shape, int dim, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            var input = CNTKLib.InputVariable(new int[] { shape }, DataType.Float);
            Function fullyConnected = FullyConnected(input, dim, useBias, weightInitializer, biasInitializer);
            return Activation(fullyConnected, activation);
        }

        /// <summary>
        /// Denses the specified layer.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="dim">Positive integer, dimensionality of the output space.</param>
        /// <param name="act">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers"/></param>
        /// <returns></returns>
        public static Function Dense(Variable layer, int dim, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            if (layer.Shape.Rank != 1)
            {
                int newDim = layer.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                layer = CNTKLib.Reshape(layer, new int[] { newDim });
            }

            Function fullyConnected = FullyConnected(layer, dim, useBias, weightInitializer, biasInitializer);
            return Activation(fullyConnected, activation);
        }

        /// <summary>
        /// Applies an activation function to an output.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations"/></param>
        /// <returns></returns>
        public static Function Activation(Variable layer, string activation = OptActivations.ReLU)
        {
            switch (activation.ToLower())
            {
                default:
                case OptActivations.None:
                    return layer;
                case OptActivations.ReLU:
                    return CNTKLib.ReLU(layer);
                case OptActivations.LeakyReLU:
                    return CNTKLib.LeakyReLU(layer);
                case OptActivations.Sigmoid:
                    return CNTKLib.Sigmoid(layer);
                case OptActivations.Tanh:
                    return CNTKLib.Tanh(layer);
                case OptActivations.Softmax:
                    return CNTKLib.Softmax(layer);
                case OptActivations.Softplus:
                    return CNTKLib.Softplus(layer);
                case OptActivations.ELU:
                    return CNTKLib.ELU(layer);
            }
        }

        /// <summary>
        /// Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="rate">A float value between 0 and 1. Fraction of the input units to drop.</param>
        /// <returns></returns>
        public static Function Dropout(Variable layer, double rate)
        {
            return CNTKLib.Dropout(layer, rate);
        }

        /// <summary>
        /// Batch normalization layer (Ioffe and Szegedy, 2014). Normalize the activations of the previous layer at each batch, i.e.applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="epsilon">Small float added to variance to avoid dividing by zero.</param>
        /// <param name="betaInitializer">Initializer for the beta weight.</param>
        /// <param name="gammaInitializers">Initializer for the gamma weight.</param>
        /// <param name="runningMeanInitializer">Initializer for the running mean weight.</param>
        /// <param name="runningStdInvInitializer">Initializer for the running standard inv weight.</param>
        /// <param name="spatial">Boolean, if yes the input data is spatial (2D). If not, then sets to 1D</param>
        /// <param name="normalizationTimeConstant">The time constant in samples of the first-order low-pass filter that is used to compute mean/variance statistics for use in inference</param>
        /// <param name="blendTimeConst">The blend time constant in samples.</param>
        /// <returns></returns>
        public static Function BatchNorm(Variable layer, float epsilon=0.001f, string betaInitializer = OptInitializers.Zeros, string gammaInitializers = OptInitializers.Ones,
                                        string runningMeanInitializer = OptInitializers.Zeros, string runningStdInvInitializer = OptInitializers.Ones, bool spatial=true, 
                                        float normalizationTimeConstant=4096f, float blendTimeConst=0.0f)
        {
            var biasParams = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(betaInitializer), GlobalParameters.Device, "");
            var scaleParams = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(gammaInitializers), GlobalParameters.Device, "");
            var runningMean = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(runningMeanInitializer), GlobalParameters.Device, "");
            var runningInvStd = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, GlobalParameters.Device);
            var runningCount = Constant.Scalar(0.0f, GlobalParameters.Device);
            bool useCudnn = false;
            if(GlobalParameters.Device.Type == DeviceKind.GPU)
            {
                useCudnn = true;
            }

            return CNTKLib.BatchNormalization(layer, scaleParams, biasParams, runningMean, runningInvStd, runningCount, spatial, normalizationTimeConstant, blendTimeConst, epsilon, useCudnn);
        }

        /// <summary>
        /// Batches the norm.
        /// </summary>
        /// <param name="shape">The input shape for batch norm layer.</param>
        /// <param name="epsilon">Small float added to variance to avoid dividing by zero.</param>
        /// <param name="betaInitializer">Initializer for the beta weight.</param>
        /// <param name="gammaInitializers">Initializer for the gamma weight.</param>
        /// <param name="runningMeanInitializer">Initializer for the running mean weight.</param>
        /// <param name="runningStdInvInitializer">Initializer for the running standard inv weight.</param>
        /// <param name="spatial">Boolean, if yes the input data is spatial (2D). If not, then sets to 1D</param>
        /// <param name="normalizationTimeConstant">The time constant in samples of the first-order low-pass filter that is used to compute mean/variance statistics for use in inference</param>
        /// <param name="blendTimeConst">The blend time constant in samples.</param>
        /// <returns></returns>
        public static Function BatchNorm(int shape, float epsilon = 0.001f, string betaInitializer = OptInitializers.Zeros, string gammaInitializers = OptInitializers.Ones,
                                       string runningMeanInitializer = OptInitializers.Zeros, string runningStdInvInitializer = OptInitializers.Ones, bool spatial = true,
                                       float normalizationTimeConstant = 4096f, float blendTimeConst = 0.0f)
        {
            var input = CNTKLib.InputVariable(new int[] { shape }, DataType.Float);
            var biasParams = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(betaInitializer), GlobalParameters.Device, "");
            var scaleParams = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(gammaInitializers), GlobalParameters.Device, "");
            var runningMean = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(runningMeanInitializer), GlobalParameters.Device, "");
            var runningInvStd = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, GlobalParameters.Device);
            var runningCount = Constant.Scalar(0.0f, GlobalParameters.Device);
            bool useCudnn = false;
            if (GlobalParameters.Device.Type == DeviceKind.GPU)
            {
                useCudnn = true;
            }

            return CNTKLib.BatchNormalization(input, scaleParams, biasParams, runningMean, runningInvStd, runningCount, spatial, normalizationTimeConstant, blendTimeConst, epsilon, useCudnn);
        }

        /// <summary>
        /// Embeddings layer
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <param name="embeddingDim">The dim.</param>
        /// <returns></returns>
        public static Function Embedding(int shape, int embeddingDim, string initializers = OptInitializers.GlorotUniform)
        {
            var input = CNTKLib.InputVariable(new int[] { shape }, true, DataType.Float);
            var embeddingParameters = new Parameter(new int[] { embeddingDim, shape }, DataType.Float, Initializers.Get(initializers), GlobalParameters.Device);
            return CNTKLib.Times(embeddingParameters, input);
        }

        /// <summary>
        /// Fully connected layer.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="outputDim">The output dim.</param>
        /// <param name="useBias">if set to <c>true</c> [use bias].</param>
        /// <param name="weightInitializer">The weight initializer.</param>
        /// <param name="biasInitializer">The bias initializer.</param>
        /// <returns></returns>
        private static Function FullyConnected(Variable input, int outputDim, bool useBias, string weightInitializer, string biasInitializer)
        {
            int inputDim = input.Shape[0];
            
            int[] s = { outputDim, inputDim };
            var weights = new Parameter(s, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);

            Parameter bias = null;
            if (useBias)
            {
                int[] s2 = { outputDim };
                bias = new Parameter(s2, DataType.Float, Initializers.Get(biasInitializer), GlobalParameters.Device);
            }
            else
            {
                int[] s2 = { outputDim };
                bias = new Parameter(s2, DataType.Float, 0.0f, GlobalParameters.Device);
            }
            
            return CNTKLib.Plus(bias, CNTKLib.Times(weights, input));
        }
    }
}
