using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public class MeanSquaredError : BaseLoss
    {
        public MeanSquaredError()
            : base("mean_squared_error")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            return Mean(Square(preds - labels));
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            var norm = 2 / preds.ElementCount();
            return (preds - labels) * norm;
        }
    }
}
