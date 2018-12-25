using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Layers;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Optimizers
{
    public class RMSProp : BaseOptimizer
    {
        public float Rho { get; set; }

        public float Epsilon { get; set; }

        private Dictionary<string, Tensor> accumulators;

        public RMSProp(float lr = 0.01f, float rho = 0.9f, float decayRate = 0, float epsilon = float.Epsilon)
            : base(lr)
        {
            DecayRate = decayRate;
            Rho = rho;
            Epsilon = epsilon;
            accumulators = new Dictionary<string, Tensor>();
        }

        public override void Update(int iteration, BaseLayer layer)
        {
            if(DecayRate > 0)
            {
                LearningRate = LearningRate * (1 / 1 + DecayRate * iteration);
            }

            foreach (var item in layer.Params)
            {
                var param = item.Value;
                if (!accumulators.ContainsKey(param.Name))
                {
                    accumulators[param.Name] = TVar.Fill(0, Global.Device, DType.Float32, param.Data.Sizes).Evaluate();
                }

                accumulators[param.Name] = (Rho * accumulators[param.Name].TVar() + (1 - Rho) * param.Grad.TVar().Pow(2)).Evaluate();

                param.Data = (param.Data.TVar() - (LearningRate * param.Grad.TVar().CDiv((accumulators[param.Name].TVar().Pow(2) + Epsilon)))).Evaluate();

                param.ApplyConstraint();
            }
        }
    }
}
