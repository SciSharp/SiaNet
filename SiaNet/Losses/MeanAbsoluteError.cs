using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class MeanAbsoluteError : BaseLoss
    {
        public MeanAbsoluteError()
            : base("mean_absolute_error")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return (preds - labels).Abs().MeanAll();
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
