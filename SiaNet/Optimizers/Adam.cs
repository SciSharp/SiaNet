namespace SiaNet.Optimizers
{
    using System;
    using System.Collections.Generic;
    using SiaNet.Engine;
    using SiaNet.Layers;

    /// <summary>
    /// Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.
    /// <para>
    /// Adam was presented by Diederik Kingma from OpenAI and Jimmy Ba from the University of Toronto in their 2015 ICLR paper(poster) titled “Adam: A Method for Stochastic Optimization“. 
    /// I will quote liberally from their paper in this post, unless stated otherwise.
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Optimizers.BaseOptimizer" />
    public class Adam : BaseOptimizer
    {
        /// <summary>
        /// Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond".
        /// </summary>
        /// <value>
        ///   <c>true</c> if [ams grad]; otherwise, <c>false</c>.
        /// </value>
        public bool AmsGrad { get; set; }

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

        /// <summary>
        /// Fuzz factor. Lowest float value but > 0
        /// </summary>
        /// <value>
        /// The epsilon.
        /// </value>
        public float Epsilon { get; set; }

        private Dictionary<string, Tensor> ms;
        private Dictionary<string, Tensor> vs;
        private Dictionary<string, Tensor> vhats;

        /// <summary>
        /// Initializes a new instance of the <see cref="Adam"/> class.
        /// </summary>
        /// <param name="lr">Initial learning rate for the optimizer</param>
        /// <param name="beta_1">The beta 1 value.</param>
        /// <param name="beta_2">The beta 2 value.</param>
        /// <param name="decayRate">Learning rate decay over each update.</param>
        /// <param name="epsilon">Fuzz factor. Lowest float value but > 0.</param>
        /// <param name="amsgrad">Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond".</param>
        public Adam(float lr = 0.01f, float beta_1 = 0.9f, float beta_2 = 0.999f, float decayRate = 0, float epsilon = 1e-08f, bool amsgrad = false)
            : base(lr, "adam")
        {
            AmsGrad = amsgrad;
            Beta1 = beta_1;
            Beta2 = beta_2;
            DecayRate = decayRate;
            Epsilon = epsilon;
            ms = new Dictionary<string, Tensor>();
            vs = new Dictionary<string, Tensor>();
            vhats = new Dictionary<string, Tensor>();
        }

        internal override void Update(int iteration, BaseLayer layer)
        {
            if (DecayRate > 0)
            {
                LearningRate = LearningRate * (1 / (1 + DecayRate * iteration));
            }

            //LearningRate = Convert.ToSingle(LearningRate * Math.Sqrt(1f - Math.Pow(Beta2, iteration)) / (1f - Math.Pow(Beta1, iteration)));
            foreach (var p in layer.Params)
            {
                var param = p.Value;
                if (!ms.ContainsKey(param.Name))
                    ms[param.Name] = K.Constant(0, param.Data.Shape);

                if (!vs.ContainsKey(param.Name))
                    vs[param.Name] = K.Constant(0, param.Data.Shape);

                if (!vhats.ContainsKey(param.Name))
                {
                    if (AmsGrad)
                        vhats[param.Name] = K.Constant(0, param.Data.Shape);
                }

                ms[param.Name] = (Beta1 * ms[param.Name]) + (1 - Beta1) * param.Grad;
                vs[param.Name] = (Beta2 * vs[param.Name]) + (1 - Beta2) * K.Square(param.Grad);
                
                var m_cap = ms[param.Name] / (1f - (float)Math.Pow(Beta1, iteration));
                var v_cap = vs[param.Name] / (1f - (float)Math.Pow(Beta2, iteration));
                //m_cap.Print();
                if (AmsGrad)
                {
                    Tensor vhat_t = K.Maximum(vhats[param.Name], v_cap);

                    param.Data = param.Data - (LearningRate * m_cap / (K.Sqrt(vhat_t) + Epsilon));
                    vhats[param.Name] = vhat_t;
                }
                else
                {
                    param.Data = param.Data - (LearningRate * m_cap / (K.Sqrt(v_cap) + Epsilon));
                }

                //param.Data.Print();
                param.ApplyConstraint();
            }
        }
    }
}