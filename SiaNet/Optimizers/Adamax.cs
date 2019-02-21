using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Engine;
using SiaNet.Layers;

namespace SiaNet.Optimizers
{
    public class Adamax : BaseOptimizer
    {
        public float Beta1 { get; set; }

        public float Beta2 { get; set; }

        private Dictionary<string, Tensor> ms;
        private Dictionary<string, Tensor> us;

        public Adamax(float lr = 0.002f, float beta_1 = 0.9f, float beta_2 = 0.999f, float decayRate = 0, float epsilon= 1e-07f)
            : base(lr, "adamax")
        {
            Beta1 = beta_1;
            Beta2 = beta_2;
            DecayRate = decayRate;
            ms = new Dictionary<string, Tensor>();
            us = new Dictionary<string, Tensor>();
        }

        public override void Update(int iteration, BaseLayer layer)
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
