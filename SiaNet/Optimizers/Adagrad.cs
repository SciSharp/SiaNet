using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Layers;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Optimizers
{
    public class Adagrad : BaseOptimizer
    {
        public float Epsilon { get; set; }

        private Dictionary<string, Tensor> accumulators;

        public Adagrad(float lr = 0.01f, float decayRate = 0, float epsilon = float.Epsilon)
            : base(lr)
        {
            DecayRate = decayRate;
            Epsilon = epsilon;
            accumulators = new Dictionary<string, Tensor>();
        }

        public override void Update(int iteration, BaseLayer layer)
        {
            if (DecayRate > 0)
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

                accumulators[param.Name] = (accumulators[param.Name].TVar() + param.Grad.TVar().Pow(2)).Evaluate();
                param.Data = (param.Data.TVar() - (LearningRate * param.Grad.TVar().CDiv(accumulators[param.Name].TVar().Sqrt() + float.Epsilon))).Evaluate();

                param.ApplyConstraint();
            }
        }
    }
}
