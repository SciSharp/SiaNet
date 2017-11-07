namespace SiaNet.Model.Optimizers
{
    using SiaNet.Common;

    /// <summary>
    /// SGD is an optimisation technique. It is an alternative to Standard Gradient Descent and other approaches like batch training or BFGS. It still leads to fast convergence, with some advantages:
    /// - Doesn't require storing all training data in memory (good for large training sets)
    /// - Allows adding new data in an "online" setting
    /// </summary>
    /// <seealso cref="SiaNet.Model.Optimizers.BaseOptimizer" />
    public class SGD : BaseOptimizer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SGD"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public SGD(double learningRate = 0.01) :
            base(OptOptimizers.SGD, learningRate)
        {

        }
    }
}
