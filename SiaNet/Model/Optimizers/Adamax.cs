namespace SiaNet.Model.Optimizers
{
    using SiaNet.Common;

    /// <summary>
    /// The Vt factor in the Adam update rule scales the gradient inversely proportionally to the ℓ2 norm of the past gradients (via the vt−1 term) and current gradient.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Optimizers.BaseOptimizer" />
    public class Adamax : BaseOptimizer
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
        /// Initializes a new instance of the <see cref="Adamax"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum.</param>
        /// <param name="varianceMomentum">The variance momentum.</param>
        /// <param name="unitGain">if set to <c>true</c> [unit gain].</param>
        /// <param name="epsilon">The epsilon.</param>
        public Adamax(double learningRate = 0.002, double momentum = 0.9, double varianceMomentum = 0.999, bool unitGain = true, double epsilon = 1e-08) :
            base(OptOptimizers.Adamax, learningRate)
        {
            Momentum = momentum;
            VarianceMomentum = varianceMomentum;
            Epsilon = epsilon;
            UnitGain = unitGain;
        }
    }
}
