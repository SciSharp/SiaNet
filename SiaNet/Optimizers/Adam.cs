using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using SiaNet.Layers;
using TensorSharp;

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
        private float lr_t = 0;

        public Adam(float lr = 0.01f, float beta_1 = 0.9f, float beta_2 = 0.999f, float decayRate = 0, float epsilon = 1e-07f, bool amsgrad = false)
            : base(lr, "adam")
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
                LearningRate = LearningRate * (1 / (1 + DecayRate * iteration));
            }

            float t = iteration;
            lr_t = LearningRate;
            //lr_t = Convert.ToSingle(LearningRate * Math.Sqrt(1f - Math.Pow(Beta2, t)) / (1f - Math.Pow(Beta1, t)));
            foreach (var p in layer.Params)
            {
                var param = p.Value;
                if (!ms.ContainsKey(param.Name))
                    ms[param.Name] = Tensor.Constant(0, Global.Device, DType.Float32, param.Data.Shape);

                if (!vs.ContainsKey(param.Name))
                    vs[param.Name] = Tensor.Constant(0, Global.Device, DType.Float32, param.Data.Shape);

                if (!vhats.ContainsKey(param.Name))
                {
                    if (AmsGrad)
                        vhats[param.Name] = Tensor.Constant(0, Global.Device, DType.Float32, param.Data.Shape);
                }

                ms[param.Name] = (Beta1 * ms[param.Name]) + (1 - Beta1) * param.Grad;
                vs[param.Name] = (Beta2 * vs[param.Name]) + (1 - Beta2) * Square(param.Grad);

                var m_cap = ms[param.Name] / (1f - (float)Math.Pow(Beta1, iteration));
                var v_cap = vs[param.Name] / (1f - (float)Math.Pow(Beta2, iteration));

                if (AmsGrad)
                {
                    Tensor vhat_t = Maximum(vhats[param.Name], v_cap);

                    param.Data = param.Data - (lr_t * m_cap / (Sqrt(vhat_t) + EPSILON));
                    vhats[param.Name] = vhat_t;
                }
                else
                {
                    param.Data = param.Data - (lr_t * m_cap / (Sqrt(v_cap) + EPSILON));
                }

                param.ApplyConstraint();
            }
        }
    }
}