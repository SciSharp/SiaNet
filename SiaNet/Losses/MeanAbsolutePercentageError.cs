using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public class MeanAbsolutePercentageError : BaseLoss
    {
        public MeanAbsolutePercentageError()
            : base("mean_absolute_percentage_error")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            var diff = Abs(preds - labels) / Clip(Abs(labels), EPSILON, float.MaxValue);
            return 100 * Mean(diff, 1).Reshape(1, -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            var diff = (preds - labels) / Clip(Abs(labels) * Abs(labels - preds), EPSILON, float.MaxValue);
            return 100 * diff / preds.Shape[0];
        }
    }
}
