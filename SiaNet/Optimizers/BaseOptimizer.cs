using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Layers;
using TensorSharp;

namespace SiaNet.Optimizers
{
    public abstract class BaseOptimizer : TOps
    {
        public float LearningRate { get; set; }

        public float Momentum { get; set; }

        public float DecayRate { get; set; }

        public BaseOptimizer(float lr)
        {
            LearningRate = lr;
        }

        public abstract void Update(int iteration, BaseLayer layer);

        public static BaseOptimizer Get(OptimizerType optimizerType)
        {
            BaseOptimizer opt = null;
            switch (optimizerType)
            {
                case OptimizerType.SGD:
                    opt = new SGD();
                    break;
                case OptimizerType.Adamax:
                    opt = new Adamax();
                    break;
                case OptimizerType.RMSprop:
                    opt = new RMSProp();
                    break;
                case OptimizerType.Adagrad:
                    opt = new Adagrad();
                    break;
                case OptimizerType.Adadelta:
                    opt = new Adadelta();
                    break;
                case OptimizerType.Adam:
                    opt = new Adam();
                    break;
                default:
                    break;
            }

            return opt;
        }
    }
}
