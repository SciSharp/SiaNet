using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class CosineProximity : BaseLoss
    {
        public CosineProximity()
            : base("cosine_proximity")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            var y_true = labels.L2Normalize();
            var y_pred = preds.L2Normalize();
            return (-1 * y_true.CMul(y_pred).SumAll());
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
