namespace SiaNet.Optimizers
{
    using System.Collections.Generic;
    using SiaNet.Engine;
    using SiaNet.Layers;

    /// <summary>
    /// Stochastic gradient descent (often shortened to SGD), also known as incremental gradient descent, is an iterative method for optimizing a differentiable objective function, 
    /// a stochastic approximation of gradient descent optimization. A 2018 article[1] implicitly credits Herbert Robbins and Sutton Monro for developing SGD in their 1951 article titled "A Stochastic Approximation Method"; see Stochastic approximation for more information. 
    /// It is called stochastic because samples are selected randomly (or shuffled) instead of as a single group (as in standard gradient descent) or in the order they appear in the training set.
    /// </summary>
    /// <seealso cref="SiaNet.Optimizers.BaseOptimizer" />
    public class SGD : BaseOptimizer
    {
        /// <summary>
        /// Whether to apply Nesterov momentum.
        /// </summary>
        /// <value>
        ///   <c>true</c> if nesterov; otherwise, <c>false</c>.
        /// </value>
        public bool Nesterov { get; set; }

        private Dictionary<string, Tensor> moments;

        /// <summary>
        /// Initializes a new instance of the <see cref="SGD"/> class.
        /// </summary>
        /// <param name="lr">The initial learning rate.</param>
        /// <param name="momentum">Parameter that accelerates SGD in the relevant direction and dampens oscillations.</param>
        /// <param name="decayRate">Learning rate decay over each update..</param>
        /// <param name="nesterov">Whether to apply Nesterov momentum.</param>
        public SGD(float lr = 0.01f, float momentum = 0, float decayRate = 0, bool nesterov = false)
            : base(lr, "sgd")
        {
            Nesterov = nesterov;
            Momentum = momentum;
            DecayRate = decayRate;
            moments = new Dictionary<string, Tensor>();
        }

        /// <summary>
        /// Updates the specified iteration.
        /// </summary>
        /// <param name="iteration">The iteration.</param>
        /// <param name="layer">The layer.</param>
        internal override void Update(int iteration, BaseLayer layer)
        {
            if (DecayRate > 0)
            {
                LearningRate = LearningRate * (1 / (1 + DecayRate * iteration));
            }

            foreach (var p in layer.Params)
            {
                Parameter param = p.Value;
                if (!moments.ContainsKey(param.Name))
                {
                    moments[param.Name] = K.Constant(0, param.Data.Shape);
                }

                moments[param.Name] = (Momentum * moments[param.Name]) - (LearningRate * param.Grad);
                if (Nesterov)
                {
                    param.Data = param.Data + (Momentum * moments[param.Name]) - (LearningRate * param.Grad);
                }
                else
                {
                    param.Data = param.Data + moments[param.Name];
                }

                param.ApplyConstraint();
            }
        }
    }
}
