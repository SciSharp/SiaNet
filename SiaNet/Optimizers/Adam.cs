using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Layers;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Optimizers
{
    public class Adam : BaseOptimizer
    {
        public bool AmsGrad { get; set; }

        public float Beta1 { get; set; }

        public float Beta2 { get; set; }

        private Dictionary<string, Tensor> ms;
        private Dictionary<string, Tensor> vs;
        private Dictionary<string, Tensor> vhats;

        public Adam(float lr = 0.01f, float beta_1 = 0.9f, float beta_2 = 0.999f, float decayRate = 0, float epsilon=float.Epsilon, bool amsgrad = false)
            : base(lr)
        {
            AmsGrad = amsgrad;
            Beta1 = beta_1;
            Beta2 = beta_2;
            DecayRate = decayRate;
            ms = new Dictionary<string, Tensor>();
            vs = new Dictionary<string, Tensor>();
            vhats = new Dictionary<string, Tensor>();
        }

        public override void Update(int iteration, BaseLayer layer)
        {
            if (DecayRate > 0)
            {
                LearningRate = LearningRate * (1 / 1 + DecayRate * iteration);
            }

            float t = iteration + 1;
            float lr_t = Convert.ToSingle(LearningRate * Math.Sqrt(1f - Math.Pow(Beta2, t)) / (1f - Math.Pow(Beta1, t)));
            foreach (var item in layer.Params)
            {
                var param = item.Value;

                if(!ms.ContainsKey(param.Name))
                {
                    ms[param.Name] = TVar.Fill(0, Global.Device, DType.Float32, param.Data.Sizes).Evaluate();
                    vs[param.Name] = TVar.Fill(0, Global.Device, DType.Float32, param.Data.Sizes).Evaluate();
                    if(AmsGrad)
                        vhats[param.Name] = TVar.Fill(0, Global.Device, DType.Float32, param.Data.Sizes).Evaluate();
                    else
                        vhats[param.Name] = TVar.Fill(0, Global.Device, DType.Float32, 1).Evaluate();

                    var m_t = (Beta1 * ms[param.Name].TVar()) + (1 - Beta1) * param.Grad.TVar();
                    var v_t = (Beta2 * vs[param.Name].TVar()) + (1 - Beta2) * param.Grad.TVar().Pow(2);

                    if (AmsGrad)
                    {
                        TVar vhat_t = TensorUtil.Maximum(vhats[param.Name], v_t);

                        param.Data = (param.Data.TVar() - lr_t * m_t.CDiv(vhat_t.Sqrt() + float.Epsilon)).Evaluate();
                        vhats[param.Name] = vhat_t.Evaluate();
                    }
                    else
                    {
                        param.Data = (param.Data.TVar() - lr_t * m_t.CDiv(v_t.Sqrt() + float.Epsilon)).Evaluate();
                    }

                    ms[param.Name] = m_t.Evaluate();
                    vs[param.Name] = v_t.Evaluate();

                    param.ApplyConstraint();
                }
            }
        }
    }
}