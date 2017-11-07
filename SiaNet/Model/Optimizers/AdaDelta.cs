namespace SiaNet.Model.Optimizers
{
    using SiaNet.Common;

    /// <summary>
    /// Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Optimizers.BaseOptimizer" />
    public class AdaDelta : BaseOptimizer
    {
        /// <summary>
        /// Gets or sets the rho.
        /// </summary>
        /// <value>
        /// The rho.
        /// </value>
        public double Rho
        {
            get
            {
                return (double)AdditionalParams["Rho"];
            }
            set
            {
                AdditionalParams["Rho"] = value;
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
        /// Initializes a new instance of the <see cref="AdaDelta"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="rho">The rho.</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <param name="regulizer">The regulizer.</param>
        public AdaDelta(double learningRate = 1.0, double rho = 0.95, double epsilon = 1e-08) :
            base(OptOptimizers.AdaDelta, learningRate)
        {
            Rho = rho;
            Epsilon = epsilon;
        }
    }
}
