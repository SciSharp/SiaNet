using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class MeanAbsolutePercentageError : BaseLoss
    {
        public MeanAbsolutePercentageError()
            : base("mean_absolute_percentage_error")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            var diff = (labels - preds).Abs().CDiv(preds.Abs());
            return (100 * diff.MeanAll());
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
