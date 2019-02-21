using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            var diff = K.Abs(preds - labels) / K.Clip(K.Abs(labels), K.Epsilon(), float.MaxValue);
            return 100 * K.Reshape(K.Mean(diff, 1), 1, -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            var diff = (preds - labels) / K.Clip(K.Abs(labels) * K.Abs(labels - preds), K.Epsilon(), float.MaxValue);
            return 100 * diff / preds.Shape[0];
        }
    }
}
