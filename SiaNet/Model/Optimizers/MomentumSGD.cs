namespace SiaNet.Model.Optimizers
{
    using SiaNet.Common;

    /// <summary>
    /// Momentum of Stochastic gradient descent optimizer.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Optimizers.BaseOptimizer" />
    public class MomentumSGD : BaseOptimizer
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
        /// Initializes a new instance of the <see cref="MomentumSGD"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum.</param>
        /// <param name="unitGain">if set to <c>true</c> [unit gain].</param>
        public MomentumSGD(double learningRate = 0.01, double momentum = 0, bool unitGain = true) :
            base(OptOptimizers.MomentumSGD, learningRate)
        {
            Momentum = momentum;
            UnitGain = unitGain;
        }
    }
}
