namespace SiaNet.Model.Optimizers
{
    using SiaNet.Common;

    /// <summary>
    /// Adagrad is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters
    /// </summary>
    /// <seealso cref="SiaNet.Model.Optimizers.BaseOptimizer" />
    public class AdaGrad : BaseOptimizer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AdaGrad"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public AdaGrad(double learningRate = 0.01) :
            base(OptOptimizers.AdaGrad, learningRate)
        {
            
        }
    }
}
