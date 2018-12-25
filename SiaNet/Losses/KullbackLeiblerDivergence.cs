using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class KullbackLeiblerDivergence : BaseLoss
    {
        public KullbackLeiblerDivergence()
            : base("kullback_leibler_divergence")
        {
        }

        public override TVar Call(TVar preds, TVar labels)
        {
            var y_true = labels.Clamp(float.Epsilon, 1);
            var y_pred = preds.Clamp(float.Epsilon, 1);

            return y_true.CMul(y_true.CDiv(y_pred).Log()).SumAll();
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
