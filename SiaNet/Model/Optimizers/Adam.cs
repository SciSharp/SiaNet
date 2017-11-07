namespace SiaNet.Model.Optimizers
{
    using SiaNet.Common;

    /// <summary>
    /// Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients mtmt, similar to momentum.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Optimizers.BaseOptimizer" />
    public class Adam : BaseOptimizer
    {
        /// <summary>
        /// Gets or sets the momentum.
        /// </summary>
        /// <value>
        /// The momentum.
        /// </value>
        public double Momentum
        {
            get
            {
                return (double)AdditionalParams["Momentum"];
            }
            set
            {
                AdditionalParams["Momentum"] = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance momentum.
        /// </summary>
        /// <value>
        /// The variance momentum.
        /// </value>
        public double VarianceMomentum
        {
            get
            {
                return (double)AdditionalParams["VarianceMomentum"];
            }
            set
            {
                AdditionalParams["VarianceMomentum"] = value;
            }
        }

        /// <summary>
        /// Gets or sets the epsilon.
        /// </summary>
        /// <value>
        /// The epsilon.
        /// </value>
        public double Epsilon
        {
            get
            {
                return (double)AdditionalParams["Epsilon"];
            }
            set
            {
                AdditionalParams["Epsilon"] = value;
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether [unit gain].
        /// </summary>
        /// <value>
        ///   <c>true</c> if [unit gain]; otherwise, <c>false</c>.
        /// </value>
        public bool UnitGain
        {
            get
            {
                return (bool)AdditionalParams["UnitGain"];
            }
            set
            {
                AdditionalParams["UnitGain"] = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Adam"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum.</param>
        /// <param name="varianceMomentum">The variance momentum.</param>
        /// <param name="unitGain">if set to <c>true</c> [unit gain].</param>
        /// <param name="epsilon">The epsilon.</param>
        public Adam(double learningRate = 0.001, double momentum = 0.9, double varianceMomentum = 0.999, bool unitGain = true, double epsilon = 1e-08f) :
            base(OptOptimizers.Adam, learningRate)
        {
            Momentum = momentum;
            VarianceMomentum = varianceMomentum;
            Epsilon = epsilon;
            UnitGain = unitGain;
        }
    }
}
