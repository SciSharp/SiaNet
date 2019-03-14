namespace SiaNet.Optimizers
{
    using System;
    using System.Collections.Generic;
    using SiaNet.Engine;
    using SiaNet.Layers;

    /// <summary>
    /// Adamax optimizer from Adam paper's Section 7.
    /// <para>
    /// It is a variant of Adam based on the infinity norm.Default parameters follow those provided in the paper.
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Optimizers.BaseOptimizer" />
    public class Adamax : BaseOptimizer
    {
        /// <summary>
        /// Gets or sets the beta 1 value.
        /// </summary>
        /// <value>
        /// The beta1.
        /// </value>
        public float Beta1 { get; set; }

        /// <summary>
        /// Gets or sets the beta 2 value.
        /// </summary>
        /// <value>
        /// The beta2.
        /// </value>
        public float Beta2 { get; set; }

        private Dictionary<string, Tensor> ms;
        private Dictionary<string, Tensor> us;

        /// <summary>
        /// Initializes a new instance of the <see cref="Adamax"/> class.
        /// </summary>
        /// <param name="lr">The initial learning rate for the optimizer.</param>
        /// <param name="beta_1">The beta 1 value.</param>
        /// <param name="beta_2">The beta 2 value.</param>
        /// <param name="decayRate">Learning rate decay over each update.</param>
        public Adamax(float lr = 0.002f, float beta_1 = 0.9f, float beta_2 = 0.999f, float decayRate = 0)
            : base(lr, "adamax")
        {
            Beta1 = beta_1;
            Beta2 = beta_2;
            DecayRate = decayRate;
            ms = new Dictionary<string, Tensor>();
            us = new Dictionary<string, Tensor>();
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

            float t = iteration + 1;
            float lr_t = Convert.ToSingle(LearningRate / (1f - Math.Pow(Beta1, t)));
            foreach (var item in layer.Params)
            {
                var param = item.Value;

                if (!ms.ContainsKey(param.Name))
                {
                    ms[param.Name] = K.Constant(0, param.Data.Shape);
                    us[param.Name] = K.Constant(0, param.Data.Shape);
                }

                var m_t = (Beta1 * ms[param.Name]) + (1 - Beta1) * param.Grad;
                var u_t = K.Maximum((Beta2 * us[param.Name]), K.Abs(param.Grad));

                param.Data = param.Data - LearningRate * m_t / (u_t + K.Epsilon());
                ms[param.Name] = m_t;
                us[param.Name] = u_t;

                param.ApplyConstraint();
            }
        }
    }
}
