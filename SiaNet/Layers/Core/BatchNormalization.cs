namespace SiaNet.Layers
{
    using SiaNet.Constraints;
    using SiaNet.Engine;
    using SiaNet.Initializers;
    using SiaNet.Regularizers;

    /// <summary>
    /// Batch normalization layer (Ioffe and Szegedy, 2014).
    /// <para>
    /// Normalize the activations of the previous layer at each batch, i.e.applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class BatchNormalization : BaseLayer
    {
        /// <summary>
        /// Integer, the axis that should be normalized (typically the features axis). For instance, after a Conv2D layer set axis=1 in BatchNormalization.
        /// </summary>
        /// <value>
        /// The axis.
        /// </value>
        public int Axis { get; set; }

        /// <summary>
        /// Momentum for the moving mean and the moving variance.
        /// </summary>
        /// <value>
        /// The momentum.
        /// </value>
        public float Momentum { get; set; }

        /// <summary>
        /// Small float added to variance to avoid dividing by zero.
        /// </summary>
        /// <value>
        /// The epsilon.
        /// </value>
        public float Epsilon { get; set; }

        /// <summary>
        /// If True, add offset of beta to normalized tensor. If False, beta is ignored.
        /// </summary>
        /// <value>
        ///   <c>true</c> if center; otherwise, <c>false</c>.
        /// </value>
        public bool Center { get; set; }

        /// <summary>
        ///  If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. relu), this can be disabled since the scaling will be done by the next layer.
        /// </summary>
        /// <value>
        ///   <c>true</c> if scale; otherwise, <c>false</c>.
        /// </value>
        public bool Scale { get; set; }

        /// <summary>
        /// Initializer for beta weight matrix
        /// </summary>
        /// <value>
        /// The beta initializer.
        /// </value>
        public BaseInitializer BetaInitializer { get; set; }

        /// <summary>
        /// Initializer for gamma weight matrix
        /// </summary>
        /// <value>
        /// The gamma initializer.
        /// </value>
        public BaseInitializer GammaInitializer { get; set; }

        /// <summary>
        /// Initializer for moving mean weight matrix
        /// </summary>
        /// <value>
        /// The moving mean initializer.
        /// </value>
        public BaseInitializer MovingMeanInitializer { get; set; }

        /// <summary>
        /// Initializer for moving variance weight matrix
        /// </summary>
        /// <value>
        /// The moving variance initializer.
        /// </value>
        public BaseInitializer MovingVarianceInitializer { get; set; }

        /// <summary>
        /// Constraint function for beta weight matrix
        /// </summary>
        /// <value>
        /// The beta constraint.
        /// </value>
        public BaseConstraint BetaConstraint { get; set; }

        /// <summary>
        /// Constraint function for gamma weight matrix
        /// </summary>
        /// <value>
        /// The gamma constraint.
        /// </value>
        public BaseConstraint GammaConstraint { get; set; }

        /// <summary>
        /// Regularizer function for beta weight matrix
        /// </summary>
        /// <value>
        /// The beta regularizer.
        /// </value>
        public BaseRegularizer BetaRegularizer { get; set; }

        /// <summary>
        /// Regularizer function for gamma weight matrix
        /// </summary>
        /// <value>
        /// The gamma regularizer.
        /// </value>
        public BaseRegularizer GammaRegularizer { get; set; }

        /// <summary>
        /// The mu parameter
        /// </summary>
        private Parameter mu;

        /// <summary>
        /// The mv parameter
        /// </summary>
        private Parameter mv;

        /// <summary>
        /// The norm
        /// </summary>
        private Tensor norm;

        /// <summary>
        /// Initializes a new instance of the <see cref="BatchNormalization"/> class.
        /// </summary>
        /// <param name="axis">Integer, the axis that should be normalized (typically the features axis). For instance, after a Conv2D layer set axis=1 in BatchNormalization.</param>
        /// <param name="momentum">Momentum for the moving mean and the moving variance.</param>
        /// <param name="epsilon">Small float added to variance to avoid dividing by zero.</param>
        /// <param name="center">If True, add offset of beta to normalized tensor. If False, beta is ignored.</param>
        /// <param name="scale">If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. relu), this can be disabled since the scaling will be done by the next layer.</param>
        /// <param name="betaInitializer">Initializer for beta weight matrix.</param>
        /// <param name="betaRegularizer">Regularizer function for beta weight matrix</param>
        /// <param name="betaConstraint">Constraint function for beta weight matrix.</param>
        /// <param name="gammaInitializer">Initializer for gamma weight matrix</param>
        /// <param name="gammaRegularizer">Regularizer function for gamma weight matrix.</param>
        /// <param name="gammaConstraint">Constraint function for gamma weight matrix.</param>
        /// <param name="movingMeanInitializer">Initializer for moving mean weight matrix.</param>
        /// <param name="movingVarianceInitializer">Initializer for moving variance weight matrix.</param>
        public BatchNormalization(int axis = -1, float momentum = 0.99f, float epsilon = 0.001f, bool center = true, bool scale = true,
                                   BaseInitializer betaInitializer = null, BaseRegularizer betaRegularizer = null, BaseConstraint betaConstraint = null, BaseInitializer gammaInitializer = null,
                                   BaseRegularizer gammaRegularizer = null, BaseConstraint gammaConstraint = null, BaseInitializer movingMeanInitializer = null, BaseInitializer movingVarianceInitializer = null)
            : base("batchnormalization")
        {
            Axis = axis;
            Momentum = momentum;
            Epsilon = epsilon;
            Center = center;
            Scale = scale;
            BetaInitializer = betaInitializer ?? new Zeros();
            GammaInitializer = gammaInitializer ?? new Ones();
            MovingMeanInitializer = movingMeanInitializer ?? new Zeros();
            MovingVarianceInitializer = movingVarianceInitializer ?? new Ones();
            BetaConstraint = betaConstraint;
            GammaConstraint = gammaConstraint;
            BetaRegularizer = betaRegularizer;
            GammaRegularizer = gammaRegularizer;
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            
            Parameter beta = BuildParam("beta", x.Shape, x.ElementType, BetaInitializer, BetaConstraint, BetaRegularizer);
            Parameter gamma = BuildParam("gamma", x.Shape, x.ElementType, GammaInitializer, GammaConstraint, GammaRegularizer);

            mu = BuildParam("mm", x.Shape, x.ElementType, MovingMeanInitializer, null, null, false);
            mv = BuildParam("mv", x.Shape, x.ElementType, MovingVarianceInitializer, null, null, false);

            norm = (x - mu.Data) / K.Sqrt((mv.Data + K.Epsilon()));
            
            var @out = gamma.Data * norm + beta.Data;
            Output = K.Reshape(@out, x.Shape);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Tensor mm = Params["mm"].Data;
            Tensor mv = Params["mv"].Data;

            var X_mu = Input.Data - mm;
            var var_inv = 1 / K.Sqrt(mv + K.Epsilon());

            var dbeta = K.Sum(outputgrad, 0);
            var dgamma = K.Sum(outputgrad * norm, 0);

            var dnorm = outputgrad * Params["gamma"].Data;
            var dvar = K.Sum(dnorm * mu.Data, 0) * K.Pow(-0.5f * (mv + K.Epsilon()), -3 / 2);
            var dmu = K.Sum(-1 * dnorm * var_inv, 0) + K.Sum(dvar * (-2 * X_mu), 0) / Input.Data.Shape[0];

            var dX = dnorm * var_inv + (dmu / Input.Data.Shape[0]) + (dvar * (2 / Input.Data.Shape[0] * X_mu));

            Input.Grad = dX;

            Params["beta"].Grad = dgamma;
            Params["gamma"].Grad = dgamma;
        }
    }
}
