using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Layers;
using TensorSharp;

namespace SiaNet.Optimizers
{
    public class Adadelta : BaseOptimizer
    {
        public float Rho { get; set; }

        public float Epsilon { get; set; }

        private Dictionary<string, Tensor> accumulators;

        private Dictionary<string, Tensor> delta_accumulators;

        public Adadelta(float lr = 1f, float rho = 0.95f, float decayRate = 0, float epsilon = float.Epsilon)
            : base(lr)
        {
            DecayRate = decayRate;
            Rho = rho;
            Epsilon = epsilon;
            accumulators = new Dictionary<string, Tensor>();
            delta_accumulators = new Dictionary<string, Tensor>();
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
                    accumulators[param.Name] = Tensor.Constant(0, Global.Device, DType.Float32, param.Data.Shape);
                    delta_accumulators[param.Name] = Tensor.Constant(0, Global.Device, DType.Float32, param.Data.Shape);
                }

                accumulators[param.Name] = (Rho * accumulators[param.Name]) + ((1 - Rho) * Square(param.Grad));
                var update = param.Grad / (Sqrt(delta_accumulators[param.Name] + float.Epsilon) / (accumulators[param.Name] + float.Epsilon));
                param.Data = param.Data - (LearningRate * update);

                param.ApplyConstraint();

                delta_accumulators[param.Name] = Rho * delta_accumulators[param.Name] + (1 - Rho) * Square(update);
            }
        }
    }
}
