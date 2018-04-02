using System.Linq;
using CNTK;
using SiaNet.Common;
using SiaNet.Model.Initializers;
using Constant = CNTK.Constant;

namespace SiaNet.NN
{
    /// <summary>
    ///     Functions to implement the basic layers like Dense, Activation, Dropout etc. They are made up of neurons that have
    ///     learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it
    ///     with a non-linearity.
    /// </summary>
    public class Basic
    {
        /// <summary>
        ///     Applies an activation function to an output.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="activation">
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie.
        ///     "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations" />
        /// </param>
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

                    return CNTKLib.LeakyReLU(layer, 0.1f);
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
                case OptActivations.SELU:

                    return CNTKLib.SELU(layer);
            }
        }

        /// <summary>
        ///     Batch normalization layer (Ioffe and Szegedy, 2014). Normalize the activations of the previous layer at each batch,
        ///     i.e.applies a transformation that maintains the mean activation close to 0 and the activation standard deviation
        ///     close to 1.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="epsilon">Small float added to variance to avoid dividing by zero.</param>
        /// <param name="betaInitializer">Initializer for the beta weight.</param>
        /// <param name="gammaInitializers">Initializer for the gamma weight.</param>
        /// <param name="runningMeanInitializer">Initializer for the running mean weight.</param>
        /// <param name="runningStdInvInitializer">Initializer for the running standard inv weight.</param>
        /// <param name="spatial">Boolean, if yes the input data is spatial (2D). If not, then sets to 1D</param>
        /// <param name="normalizationTimeConstant">
        ///     The time constant in samples of the first-order low-pass filter that is used to
        ///     compute mean/variance statistics for use in inference
        /// </param>
        /// <param name="blendTimeConst">The blend time constant in samples.</param>
        /// <returns></returns>
        public static Function BatchNorm(
            Variable layer,
            float epsilon = 0.001f,
            InitializerBase betaInitializer = null,
            InitializerBase gammaInitializers = null,
            InitializerBase runningMeanInitializer = null,
            InitializerBase runningStdInvInitializer = null,
            bool spatial = true,
            float normalizationTimeConstant = 4096f,
            float blendTimeConst = 0.0f)
        {
            betaInitializer = betaInitializer ?? new Zeros();
            gammaInitializers = gammaInitializers ?? new Zeros();
            runningMeanInitializer = runningMeanInitializer ?? new Zeros();
            runningStdInvInitializer = runningStdInvInitializer ?? new Zeros();

            var biasParams = new Parameter(new[] {NDShape.InferredDimension}, DataType.Float, betaInitializer.ToDictionary(),
                GlobalParameters.Device, "");
            var scaleParams = new Parameter(new[] {NDShape.InferredDimension}, DataType.Float, gammaInitializers.ToDictionary(),
                GlobalParameters.Device, "");
            var runningMean = new Parameter(new[] {NDShape.InferredDimension}, DataType.Float,
                runningMeanInitializer.ToDictionary(), GlobalParameters.Device, "");
            var runningInvStd = new Constant(new[] {NDShape.InferredDimension}, 0.0f, GlobalParameters.Device);
            var runningCount = Constant.Scalar(0.0f, GlobalParameters.Device);
            var useCudnn = false;

            if (GlobalParameters.Device.Type == DeviceKind.GPU)
            {
                useCudnn = true;
            }

            return CNTKLib.BatchNormalization(layer, scaleParams, biasParams, runningMean, runningInvStd, runningCount,
                spatial, normalizationTimeConstant, blendTimeConst, epsilon, useCudnn);
        }

        /// <summary>
        ///     Batches the norm.
        /// </summary>
        /// <param name="shape">The input shape for batch norm layer.</param>
        /// <param name="epsilon">Small float added to variance to avoid dividing by zero.</param>
        /// <param name="betaInitializer">Initializer for the beta weight.</param>
        /// <param name="gammaInitializers">Initializer for the gamma weight.</param>
        /// <param name="runningMeanInitializer">Initializer for the running mean weight.</param>
        /// <param name="runningStdInvInitializer">Initializer for the running standard inv weight.</param>
        /// <param name="spatial">Boolean, if yes the input data is spatial (2D). If not, then sets to 1D</param>
        /// <param name="normalizationTimeConstant">
        ///     The time constant in samples of the first-order low-pass filter that is used to
        ///     compute mean/variance statistics for use in inference
        /// </param>
        /// <param name="blendTimeConst">The blend time constant in samples.</param>
        /// <returns></returns>
        public static Function BatchNorm(
            int shape,
            float epsilon = 0.001f,
            InitializerBase betaInitializer = null,
            InitializerBase gammaInitializers = null,
            InitializerBase runningMeanInitializer = null,
            InitializerBase runningStdInvInitializer = null,
            bool spatial = true,
            float normalizationTimeConstant = 4096f,
            float blendTimeConst = 0.0f)
        {
            betaInitializer = betaInitializer ?? new Zeros();
            gammaInitializers = gammaInitializers ?? new Zeros();
            runningMeanInitializer = runningMeanInitializer ?? new Zeros();
            runningStdInvInitializer = runningStdInvInitializer ?? new Zeros();

            var input = CNTKLib.InputVariable(new[] {shape}, DataType.Float);
            var biasParams = new Parameter(new[] {NDShape.InferredDimension}, DataType.Float, betaInitializer.ToDictionary(),
                GlobalParameters.Device, "");
            var scaleParams = new Parameter(new[] {NDShape.InferredDimension}, DataType.Float, gammaInitializers.ToDictionary(),
                GlobalParameters.Device, "");
            var runningMean = new Parameter(new[] {NDShape.InferredDimension}, DataType.Float,
                runningMeanInitializer.ToDictionary(), GlobalParameters.Device, "");
            var runningInvStd = new Constant(new[] {NDShape.InferredDimension}, 0.0f, GlobalParameters.Device);
            var runningCount = Constant.Scalar(0.0f, GlobalParameters.Device);
            var useCudnn = GlobalParameters.Device.Type == DeviceKind.GPU;

            return CNTKLib.BatchNormalization(input, scaleParams, biasParams, runningMean, runningInvStd, runningCount,
                spatial, normalizationTimeConstant, blendTimeConst, epsilon, useCudnn);
        }

        /// <summary>
        ///     Denses the specified layer.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="dim">Positive integer, dimensionality of the output space.</param>
        /// <param name="act">
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie. "linear"
        ///     activation: a(x) = x). <see cref="SiaNet.Common.OptActivations" />
        /// </param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">
        ///     Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers" />
        /// </param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers" /></param>
        /// <returns></returns>
        public static Function Dense(
            Variable layer,
            int dim,
            string activation = OptActivations.None,
            bool useBias = false,
            InitializerBase weightInitializer = null,
            InitializerBase biasInitializer = null)
        {
            weightInitializer = weightInitializer ?? new Xavier();
            biasInitializer = biasInitializer ?? new Zeros();

            if (layer.Shape.Rank != 1)
            {
                var newDim = layer.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                layer = CNTKLib.Reshape(layer, new[] {newDim});
            }

            var fullyConnected = FullyConnected(layer, dim, useBias, weightInitializer, biasInitializer);

            return Activation(fullyConnected, activation);
        }

        /// <summary>
        ///     Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which
        ///     helps prevent overfitting.
        /// </summary>
        /// <param name="layer">The output of the last layer.</param>
        /// <param name="rate">A float value between 0 and 1. Fraction of the input units to drop.</param>
        /// <returns></returns>
        public static Function Dropout(Variable layer, double rate)
        {
            return CNTKLib.Dropout(layer, rate);
        }

        /// <summary>
        ///     Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]].
        ///     This layer can only be used as the first layer in a model.
        /// </summary>
        /// <param name="shape">The shape of the input.</param>
        /// <param name="embeddingDim">The dim of the output space.</param>
        /// <param name="initializers">The weight initializers for this layer.</param>
        /// <returns></returns>
        public static Function Embedding(int shape, int embeddingDim, InitializerBase initializers = null)
        {
            initializers = initializers ?? new GlorotUniform();

            var input = CNTKLib.InputVariable(new[] {shape}, true, DataType.Float);
            var embeddingParameters = new Parameter(new[] {embeddingDim, shape}, DataType.Float, initializers.ToDictionary(),
                GlobalParameters.Device);

            return CNTKLib.Times(embeddingParameters, input);
        }

        /// <summary>
        ///     Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -&gt; [[0.25, 0.1], [0.6,
        ///     -0.2]]. This layer can only be used as the first layer in a model.
        /// </summary>
        /// <param name="shape">The shape of the input.</param>
        /// <param name="embeddingDim">The dim of the output space.</param>
        /// <param name="initializers">The weight initializers for this layer.</param>
        /// <returns></returns>
        public static Function Embedding(Variable layer, int embeddingDim, InitializerBase initializers = null)
        {
            if (initializers == null)
            {
                initializers = new GlorotUniform();
            }

            var embeddingParameters = new Parameter(new[] {embeddingDim, layer.Shape[0]}, DataType.Float,
                initializers.ToDictionary(), GlobalParameters.Device);

            return CNTKLib.Times(embeddingParameters, layer);
        }

        /// <summary>
        ///     Reshapes an output to a certain shape.
        /// </summary>
        /// <param name="layer">The input layer to be reshaped.</param>
        /// <param name="targetShape">List of integers. Does not include the batch axis.</param>
        /// <returns></returns>
        public static Function Reshape(Variable layer, int[] targetShape)
        {
            return CNTKLib.Reshape(layer, targetShape);
        }

        /// <summary>
        ///     Reshapes an output to a certain shape.
        /// </summary>
        /// <param name="shape">The input shape of the data.</param>
        /// <param name="targetShape">List of integers. Does not include the batch axis.</param>
        /// <returns></returns>
        public static Function Reshape(int[] shape, int[] targetShape)
        {
            return CNTKLib.Reshape(Variable.InputVariable(shape, DataType.Float), targetShape);
        }


        /// <summary>
        ///     Fullies the connected.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="outputDim">The output dim.</param>
        /// <param name="useBias">if set to <c>true</c> [use bias].</param>
        /// <param name="weightInitializer">The weight initializer.</param>
        /// <param name="biasInitializer">The bias initializer.</param>
        /// <returns></returns>
        private static Function FullyConnected(
            Variable input,
            int outputDim,
            bool useBias,
            InitializerBase weightInitializer,
            InitializerBase biasInitializer)
        {
            var inputDim = input.Shape[0];

            int[] s = {outputDim, inputDim};
            var weights = new Parameter(s, DataType.Float, weightInitializer.ToDictionary(), GlobalParameters.Device);

            Parameter bias = null;

            if (useBias)
            {
                int[] s2 = {outputDim};
                bias = new Parameter(s2, DataType.Float, biasInitializer.ToDictionary(), GlobalParameters.Device);
            }
            else
            {
                int[] s2 = {outputDim};
                bias = new Parameter(s2, DataType.Float, 0.0f, GlobalParameters.Device);
            }

            return CNTKLib.Plus(bias, CNTKLib.Times(weights, input));
        }
    }
}