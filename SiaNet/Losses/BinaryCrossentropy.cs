using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public class BinaryCrossentropy : BaseLoss
    {
        public bool FromLogit { get; set; }

        public BinaryCrossentropy(bool fromLogit = false)
            : base("binary_crossentropy")
        {
            FromLogit = fromLogit;
        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            Tensor output = null;
            if(!FromLogit)
            {
                output = Clip(labels, EPSILON, 1f - EPSILON);
                output = Log(output / (1 - output));
            }

            return preds * Log(-1 * Sigmoid(output)) + (1 - preds) * Log(-1 * (1 - Sigmoid(output)));
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            throw new NotImplementedException();
        }
    }
}
