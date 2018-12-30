using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public class MeanAbsoluteError : BaseLoss
    {
        public MeanAbsoluteError()
            : base("mean_absolute_error")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            return Mean(Abs(preds - labels));
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            throw new NotImplementedException();
        }
    }
}
