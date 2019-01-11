using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public class KullbackLeiblerDivergence : BaseLoss
    {
        public KullbackLeiblerDivergence()
            : base("kullback_leibler_divergence")
        {
        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            var y_true = Clip(labels, EPSILON, 1);
            var y_pred = Clip(preds, EPSILON, 1);

            return Sum(y_true * Log(y_true / y_pred), -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            var y_true = Clip(labels, EPSILON, 1);
            var y_pred = Clip(preds, EPSILON, 1);

            return Maximum((-1 * (y_true / y_pred)), 0);
        }
    }
}
