namespace SiaNet.Optimizers
{
    using System.Collections.Generic;
    using SiaNet.Engine;
    using SiaNet.Layers;

    /// <summary>
    /// Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done. Compared to Adagrad, in the original version of Adadelta you don't have to set an initial learning rate. In this version, initial learning rate and decay factor can be set, as in most other optimizers.
    /// </summary>
    /// <seealso cref="SiaNet.Optimizers.BaseOptimizer" />
    public class Adadelta : BaseOptimizer
    {
        /// <summary>
        /// Adadelta decay factor, corresponding to fraction of gradient to keep at each time step.
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

        private Dictionary<string, Tensor> delta_accumulators;

        /// <summary>
        /// Initializes a new instance of the <see cref="Adadelta"/> class.
        /// </summary>
        /// <param name="lr">Initial learning rate, defaults to 1. It is recommended to leave it at the default value.</param>
        /// <param name="rho">Adadelta decay factor, corresponding to fraction of gradient to keep at each time step.</param>
        /// <param name="decayRate">Learning rate decay factor over each update.</param>
        /// <param name="epsilon">The epsilon.</param>
        public Adadelta(float lr = 1f, float rho = 0.95f, float decayRate = 0, float epsilon = 1e-07f)
            : base(lr, "adadelta")
        {
            DecayRate = decayRate;
            Rho = rho;
            Epsilon = epsilon;
            accumulators = new Dictionary<string, Tensor>();
            delta_accumulators = new Dictionary<string, Tensor>();
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

            foreach (var item in layer.Params)
            {
                var param = item.Value;
                if (!accumulators.ContainsKey(param.Name))
                {
                    accumulators[param.Name] = K.Constant(0, param.Data.Shape);
                    delta_accumulators[param.Name] = K.Constant(0, param.Data.Shape);
                }

                accumulators[param.Name] = (Rho * accumulators[param.Name]) + ((1 - Rho) * K.Square(param.Grad));
                var update = param.Grad * K.Sqrt(delta_accumulators[param.Name] + K.Epsilon()) / K.Sqrt(accumulators[param.Name] + K.Epsilon());
                param.Data = param.Data - (LearningRate * update);

                param.ApplyConstraint();

                delta_accumulators[param.Name] = Rho * delta_accumulators[param.Name] + (1 - Rho) * K.Square(update);
            }
        }
    }
}
