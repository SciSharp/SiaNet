using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Layers;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Optimizers
{
    public class SGD : BaseOptimizer
    {
        public bool Nesterov { get; set; }

        private Dictionary<string, Tensor> velocity;

        public SGD(float lr = 0.01f, float momentum = 0.9f, float decayRate = 0, bool nesterov = false)
            : base(lr)
        {
            Nesterov = nesterov;
            Momentum = momentum;
            DecayRate = decayRate;
            velocity = new Dictionary<string, Tensor>();
        }

        public override void Update(int iteration, BaseLayer layer)
        {
            if (DecayRate > 0)
            {
                LearningRate = LearningRate * (1 / 1 + DecayRate * iteration);
            }

            foreach (var p in layer.Params)
            {
                Variable param = p.Value;
                if (!velocity.ContainsKey(param.Name))
                {
                    velocity[param.Name] = TVar.Fill(0, Global.Device, DType.Float32, param.Data.Sizes).Evaluate();
                }

                velocity[param.Name] = ((velocity[param.Name].TVar() * Momentum) - (LearningRate * param.Grad.TVar())).Evaluate();
                if (Nesterov)
                {
                    param.Data = (param.Data.TVar() + (Momentum * velocity[param.Name].TVar()) - (LearningRate * param.Grad.TVar())).Evaluate();
                }
                else
                {
                    param.Data = (param.Data.TVar() - velocity[param.Name]).Evaluate();
                }

                param.ApplyConstraint();
            }
        }
    }
}
