namespace SiaNet.Optimizers
{
    using System.Collections.Generic;
    using SiaNet.Engine;
    using SiaNet.Layers;

    /// <summary>
    /// Nesterov Adam optimizer.
    /// <para>
    /// Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum. 
    /// Default parameters follow those provided in the paper.It is recommended to leave the parameters of this optimizer at their default values.
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Optimizers.BaseOptimizer" />
    public class RMSProp : BaseOptimizer
    {
        /// <summary>
        /// RMSProp decay factor, corresponding to fraction of gradient to keep at each time step.
        /// </summary>
        /// <value>
        /// The rho.
        /// </value>
        public float Rho { get; set; }

        /// <summary>
        /// Fuzz factor. Lowest float value but > 0
        /// </summary>
        /// <value>
        /// The epsilon.
        /// </value>
        public float Epsilon { get; set; }

        private Dictionary<string, Tensor> accumulators;

        /// <summary>
        /// Initializes a new instance of the <see cref="RMSProp"/> class.
        /// </summary>
        /// <param name="lr">The initial learning rate for the optimizer.</param>
        /// <param name="rho">RMSProp decay factor, corresponding to fraction of gradient to keep at each time step.</param>
        /// <param name="decayRate">Learning rate decay over each update.</param>
        /// <param name="epsilon">Fuzz factor. Lowest float value but > 0</param>
        public RMSProp(float lr = 0.001f, float rho = 0.9f, float decayRate = 0, float epsilon = 1e-07f)
            : base(lr, "rmsprop")
        {
            DecayRate = decayRate;
            Rho = rho;
            Epsilon = epsilon;
            accumulators = new Dictionary<string, Tensor>();
        }

        /// <summary>
        /// Updates the specified iteration.
        /// </summary>
        /// <param name="iteration">The iteration.</param>
        /// <param name="layer">The layer.</param>
        internal override void Update(int iteration, BaseLayer layer)
        {
            if(DecayRate > 0)
            {
                LearningRate = LearningRate * (1 / (1 + DecayRate * iteration));
            }

            foreach (var item in layer.Params)
            {
                var param = item.Value;
                if (!accumulators.ContainsKey(param.Name))
                {
                    accumulators[param.Name] = K.Constant(0, param.Data.Shape);
                }

                accumulators[param.Name] = Rho * accumulators[param.Name] + (1 - Rho) * K.Square(param.Grad);

                param.Data = param.Data - (LearningRate * param.Grad / (K.Sqrt(accumulators[param.Name]) + Epsilon));

                param.ApplyConstraint();
            }
        }
    }
}
