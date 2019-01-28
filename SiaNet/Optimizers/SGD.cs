using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Layers;
using TensorSharp;

namespace SiaNet.Optimizers
{
    public class SGD : BaseOptimizer
    {
        public bool Nesterov { get; set; }

        private Dictionary<string, Tensor> moments;

        public SGD(float lr = 0.01f, float momentum = 0, float decayRate = 0, bool nesterov = false)
            : base(lr, "sgd")
        {
            Nesterov = nesterov;
            Momentum = momentum;
            DecayRate = decayRate;
            moments = new Dictionary<string, Tensor>();
        }

        public override void Update(int iteration, BaseLayer layer)
        {
            if (DecayRate > 0)
            {
                LearningRate = LearningRate * (1 / (1 + DecayRate * iteration));
            }

            foreach (var p in layer.Params)
            {
                Parameter param = p.Value;
                if (!moments.ContainsKey(param.Name))
                {
                    moments[param.Name] = Tensor.Constant(0, Global.Device, DType.Float32, param.Data.Shape);
                }

                moments[param.Name] = (Momentum * moments[param.Name]) - (LearningRate * param.Grad);
                if (Nesterov)
                {
                    param.Data = param.Data + (Momentum * moments[param.Name]) - (LearningRate * param.Grad);
                }
                else
                {
                    param.Data = param.Data + moments[param.Name];
                }

                param.ApplyConstraint();
            }
        }
    }
}
