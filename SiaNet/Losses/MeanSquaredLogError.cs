using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class MeanSquaredLogError : BaseLoss
    {
        public MeanSquaredLogError()
            : base("mean_squared_logarithmic_error")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            var first_log = (preds + 1).Log();
            var second_log = (labels + 1).Log();

            return (first_log - second_log).Pow(2).MeanAll();
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
