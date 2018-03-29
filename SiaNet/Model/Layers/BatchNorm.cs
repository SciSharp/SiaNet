using Newtonsoft.Json;
using SiaNet.Common;
using SiaNet.Model.Initializers;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Batch normalization layer (Ioffe and Szegedy, 2014). Normalize the activations of the previous layer at each batch,
    ///     i.e.applies a transformation that maintains the mean activation close to 0 and the activation standard deviation
    ///     close to 1.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class BatchNorm : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="BatchNorm" /> class.
        /// </summary>
        /// <param name="shape">The input shape for batch norm layer.</param>
        public BatchNorm(int shape)
            : this()
        {
            Shape = shape;
        }
        
        /// <summary>
        ///     Initializes a new instance of the <see cref="T:SiaNet.Model.Layers.BatchNorm" /> class.
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
            string betaInitializer = OptInitializers.Zeros,
            string gammaInitializer = OptInitializers.Ones,
            string runningMeanInitializer = OptInitializers.Zeros,
            string runningStdInvInitializer = OptInitializers.Ones,
            bool spatial = true,
            float normalizationTimeConstant = 4096f,
            float blendTimeConst = 0.0f,
            float epsilon = 0.001f)
            : this()
        {
            Shape = null;
            Epsilon = epsilon;
            BetaInitializer = new Initializer(betaInitializer);
            GammaInitializer = new Initializer(gammaInitializer);
            RunningMeanInitializer = new Initializer(runningMeanInitializer);
            RunningStdInvInitializer = new Initializer(runningStdInvInitializer);
            Spatial = spatial;
            NormalizationTimeConstant = normalizationTimeConstant;
            BlendTimeConst = blendTimeConst;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="BatchNorm" /> class.
        /// </summary>
        /// <param name="shape">The input shape for batch norm layer.</param>
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
            int shape,
            string betaInitializer = OptInitializers.Zeros,
            string gammaInitializer = OptInitializers.Ones,
            string runningMeanInitializer = OptInitializers.Zeros,
            string runningStdInvInitializer = OptInitializers.Ones,
            bool spatial = true,
            float normalizationTimeConstant = 4096f,
            float blendTimeConst = 0.0f,
            float epsilon = 0.001f)
            : this(betaInitializer, gammaInitializer, runningMeanInitializer, runningStdInvInitializer, spatial,
                normalizationTimeConstant, blendTimeConst, epsilon)
        {
            Shape = shape;
        }

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
            Initializer betaInitializer = null,
            Initializer gammaInitializer = null,
            Initializer runningMeanInitializer = null,
            Initializer runningStdInvInitializer = null,
            bool spatial = true,
            float normalizationTimeConstant = 4096f,
            float blendTimeConst = 0.0f,
            float epsilon = 0.001f)
            : this()
        {
            Shape = null;
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
        ///     Initializes a new instance of the <see cref="BatchNorm" /> class.
        /// </summary>
        /// <param name="shape">The input shape for batch norm layer.</param>
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
            int shape,
            Initializer betaInitializer = null,
            Initializer gammaInitializer = null,
            Initializer runningMeanInitializer = null,
            Initializer runningStdInvInitializer = null,
            bool spatial = true,
            float normalizationTimeConstant = 4096f,
            float blendTimeConst = 0.0f,
            float epsilon = 0.001f)
            : this(betaInitializer, gammaInitializer, runningMeanInitializer, runningStdInvInitializer, spatial,
                normalizationTimeConstant, blendTimeConst, epsilon)
        {
            Shape = shape;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="BatchNorm" /> class.
        /// </summary>
        internal BatchNorm()
        {
            Name = "BatchNorm";
            Shape = null;
            BetaInitializer = new Zeros();
            GammaInitializer = new Ones();
            RunningMeanInitializer = new Zeros();
            RunningStdInvInitializer = new Ones();
            Spatial = true;
            NormalizationTimeConstant = 4096f;
            BlendTimeConst = 0;
            Epsilon = 0.001f;
        }

        /// <summary>
        ///     Initializer for the beta weight.
        /// </summary>
        /// <value>
        ///     The beta initializer.
        /// </value>
        [JsonIgnore]
        public Initializer BetaInitializer
        {
            get => GetParam<Initializer>("BetaInitializer");

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
        public Initializer GammaInitializer
        {
            get => GetParam<Initializer>("GammaInitializer");

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
        public Initializer RunningMeanInitializer
        {
            get => GetParam<Initializer>("RunningMeanInitializer");

            set => SetParam("RunningMeanInitializer", value);
        }

        /// <summary>
        ///     Initializer for the running standard inv weight.
        /// </summary>
        /// <value>
        ///     The running standard inv initializer.
        /// </value>
        [JsonIgnore]
        public Initializer RunningStdInvInitializer
        {
            get => GetParam<Initializer>("RunningStdInvInitializer");

            set => SetParam("RunningStdInvInitializer", value);
        }

        /// <summary>
        ///     The input shape for batch norm layer
        /// </summary>
        /// <value>
        ///     The shape.
        /// </value>
        [JsonIgnore]
        public int? Shape
        {
            get => GetParam<int?>("Shape");

            set => SetParam("Shape", value);
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
    }
}