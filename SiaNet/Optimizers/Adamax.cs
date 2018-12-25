using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Layers;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Optimizers
{
    public class Adamax : BaseOptimizer
    {
        public float Beta1 { get; set; }

        public float Beta2 { get; set; }

        private Dictionary<string, Tensor> ms;
        private Dictionary<string, Tensor> us;

        public Adamax(float lr = 0.002f, float beta_1 = 0.9f, float beta_2 = 0.999f, float decayRate = 0, float epsilon=float.Epsilon)
            : base(lr)
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
                LearningRate = LearningRate * (1 / 1 + DecayRate * iteration);
            }

            float t = iteration + 1;
            float lr_t = Convert.ToSingle(LearningRate / Math.Sqrt(1f - Math.Pow(Beta1, t)));
            foreach (var item in layer.Params)
            {
                var param = item.Value;

                if (!ms.ContainsKey(param.Name))
                {
                    ms[param.Name] = TVar.Fill(0, Global.Device, DType.Float32, param.Data.Sizes).Evaluate();
                    us[param.Name] = TVar.Fill(0, Global.Device, DType.Float32, param.Data.Sizes).Evaluate();

                    TVar m_t = (Beta1 * ms[param.Name].TVar()) + (1 - Beta1) * param.Grad.TVar();
                    TVar u_t = TensorUtil.Maximum((Beta2 * us[param.Name].TVar()), param.Grad.TVar().Abs());

                    param.Data = (param.Data - lr_t * m_t.CDiv(u_t + float.Epsilon)).Evaluate();
                    ms[param.Name] = m_t.Evaluate();
                    us[param.Name] = u_t.Evaluate();

                    param.ApplyConstraint();
                }
            }
        }
    }
}
