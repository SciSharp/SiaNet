using CNTK;
using Newtonsoft.Json;
using SiaNet.Model.Initializers;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Batch normalization layer (Ioffe and Szegedy, 2014). Normalize the activations of the previous layer at each batch,
    ///     i.e.applies a transformation that maintains the mean activation close to 0 and the activation standard deviation
    ///     close to 1.
    /// </summary>
    /// <seealso cref="OptimizableLayerBase" />
    public class BatchNorm : OptimizableLayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="BatchNorm" /> class.
        /// </summary>
        /// <param name="epsilon">Small float added to variance to avoid dividing by zero.</param>
        /// <param name="betaInitializer">Initializer for the beta weight.</param>
        /// <param name="gammaInitializer">Initializer for the gamma weight.</param>
        /// <param name="runningMeanInitializer">Initializer for the running mean weight.</param>
        /// <param name="runningStdInvInitializer">Initializer for the running standard inv weight.</param>
        /// <param name="spatial">Boolean, if yes the input data is spatial (2D). If not, then sets to 1D</param>
        /// <param name="normalizationTimeConstant">
        ///     The time constant in samples of the first-order low-pass filter that is used to
        ///     compute mean/variance statistics for use in inference
        /// </param>
        /// <param name="blendTimeConst">The blend time constant in samples.</param>
        public BatchNorm(
            InitializerBase betaInitializer = null,
            InitializerBase gammaInitializer = null,
            InitializerBase runningMeanInitializer = null,
            InitializerBase runningStdInvInitializer = null,
            bool spatial = true,
            float normalizationTimeConstant = 4096f,
            float blendTimeConst = 0.0f,
            float epsilon = 0.001f)
        {
            Epsilon = epsilon;
            BetaInitializer = betaInitializer ?? new Zeros();
            GammaInitializer = gammaInitializer ?? new Ones();
            RunningMeanInitializer = runningMeanInitializer ?? new Zeros();
            RunningStdInvInitializer = runningStdInvInitializer ?? new Ones();
            Spatial = spatial;
            NormalizationTimeConstant = normalizationTimeConstant;
            BlendTimeConst = blendTimeConst;
        }


        /// <summary>
        ///     Initializer for the beta weight.
        /// </summary>
        /// <value>
        ///     The beta initializer.
        /// </value>
        [JsonIgnore]
        public InitializerBase BetaInitializer
        {
            get => GetParam<InitializerBase>("BetaInitializer");

            set => SetParam("BetaInitializer", value);
        }

        /// <summary>
        ///     The blend time constant in samples.
        /// </summary>
        /// <value>
        ///     The blend time constant.
        /// </value>
        [JsonIgnore]
        public float BlendTimeConst
        {
            get => GetParam<float>("BlendTimeConst");

            set => SetParam("BlendTimeConst", value);
        }

        /// <summary>
        ///     Small float added to variance to avoid dividing by zero.
        /// </summary>
        /// <value>
        ///     The epsilon.
        /// </value>
        [JsonIgnore]
        public float Epsilon
        {
            get => GetParam<float>("Epsilon");

            set => SetParam("Epsilon", value);
        }

        /// <summary>
        ///     Initializer for the gamma weight.
        /// </summary>
        /// <value>
        ///     The gamma initializer.
        /// </value>
        [JsonIgnore]
        public InitializerBase GammaInitializer
        {
            get => GetParam<InitializerBase>("GammaInitializer");

            set => SetParam("GammaInitializer", value);
        }

        /// <summary>
        ///     The time constant in samples of the first-order low-pass filter that is used to compute mean/variance statistics
        ///     for use in inference
        /// </summary>
        /// <value>
        ///     The normalization time constant.
        /// </value>
        [JsonIgnore]
        public float NormalizationTimeConstant
        {
            get => GetParam<float>("NormalizationTimeConstant");

            set => SetParam("NormalizationTimeConstant", value);
        }

        /// <summary>
        ///     Initializer for the running mean weight.
        /// </summary>
        /// <value>
        ///     The running mean initializer.
        /// </value>
        [JsonIgnore]
        public InitializerBase RunningMeanInitializer
        {
            get => GetParam<InitializerBase>("RunningMeanInitializer");

            set => SetParam("RunningMeanInitializer", value);
        }

        /// <summary>
        ///     Initializer for the running standard inv weight.
        /// </summary>
        /// <value>
        ///     The running standard inv initializer.
        /// </value>
        [JsonIgnore]
        public InitializerBase RunningStdInvInitializer
        {
            get => GetParam<InitializerBase>("RunningStdInvInitializer");

            set => SetParam("RunningStdInvInitializer", value);
        }

        /// <summary>
        ///     Boolean, if yes the input data is spatial (2D). If not, then sets to 1D
        /// </summary>
        /// <value>
        ///     <c>true</c> if spatial; otherwise, <c>false</c>.
        /// </value>
        [JsonIgnore]
        public bool Spatial
        {
            get => GetParam<bool>("Spatial");

            set => SetParam("Spatial", value);
        }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            //if (inputFunction.Shape.Rank != 1)
            //{
            //    throw new ArgumentException("Variable has an invalid shape.", nameof(inputFunction));
            //}

            var biasParams = new CNTK.Parameter(new[] {NDShape.InferredDimension}, DataType.Float,
                BetaInitializer.ToDictionary(),
                GlobalParameters.Device, "");
            var scaleParams = new CNTK.Parameter(new[] {NDShape.InferredDimension}, DataType.Float,
                GammaInitializer.ToDictionary(),
                GlobalParameters.Device, "");
            var runningMean = new CNTK.Parameter(new[] {NDShape.InferredDimension}, DataType.Float,
                RunningMeanInitializer.ToDictionary(), GlobalParameters.Device, "");
            var runningInvStd = new CNTK.Constant(new[] {NDShape.InferredDimension}, 0.0f, GlobalParameters.Device);
            var runningCount = CNTK.Constant.Scalar(0.0f, GlobalParameters.Device);
            var useCuda = GlobalParameters.Device.Type == DeviceKind.GPU;

            return CNTKLib.BatchNormalization(inputFunction, scaleParams, biasParams, runningMean, runningInvStd,
                runningCount,
                Spatial, NormalizationTimeConstant, BlendTimeConst, Epsilon, useCuda);
        }
    }
}