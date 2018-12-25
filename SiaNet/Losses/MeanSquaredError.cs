using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class MeanSquaredError : BaseLoss
    {
        public MeanSquaredError()
            : base("mean_squared_error")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return (preds - labels).Pow(2).MeanAll();
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            var norm = 2.0f / preds.Evaluate().ElementCount();

            return ((preds - labels) * norm);
        }
    }
}
