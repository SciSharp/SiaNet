using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Engine;
using SiaNet.Layers;

namespace SiaNet.Optimizers
{
    public class Adadelta : BaseOptimizer
    {
        public float Rho { get; set; }

        public float Epsilon { get; set; }

        private Dictionary<string, Tensor> accumulators;

        private Dictionary<string, Tensor> delta_accumulators;

        public Adadelta(float lr = 1f, float rho = 0.95f, float decayRate = 0, float epsilon = 1e-07f)
            : base(lr, "adadelta")
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
                LearningRate = LearningRate * (1 / (1 + DecayRate * iteration));
            }

            foreach (var item in layer.Params)
            {
                var param = item.Value;
                if (!accumulators.ContainsKey(param.Name))
                {
                    accumulators[param.Name] = K.Constant(0, param.Data.Shape);
                    delta_accumulators[param.Name] = K.Constant(0, param.Data.Shape);
                }

                accumulators[param.Name] = (Rho * accumulators[param.Name]) + ((1 - Rho) * K.Square(param.Grad));
                var update = param.Grad * K.Sqrt(delta_accumulators[param.Name] + K.Epsilon()) / K.Sqrt(accumulators[param.Name] + K.Epsilon());
                param.Data = param.Data - (LearningRate * update);

                param.ApplyConstraint();

                delta_accumulators[param.Name] = Rho * delta_accumulators[param.Name] + (1 - Rho) * K.Square(update);
            }
        }
    }
}
