using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
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
            Tensor output = preds;
            if (!FromLogit)
            {
                output = Clip(output, EPSILON, 1f - EPSILON);
                output = Log(output / (1 - output));
            }

            float scale = (2f * preds.ElementCount()) / 3f;
            output = Sigmoid(output);

            return -1 * labels * Log(output) - (1 - labels) * Log(1 - output);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            Tensor output = Clip(preds, EPSILON, 1f - EPSILON);
            return -1 * (labels - 1) / (1 - output) - labels / output;
        }
    }
}
