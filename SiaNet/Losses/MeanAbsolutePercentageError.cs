using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

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
            var diff = Abs(labels - preds) / Abs(preds);
            return 100 * Mean(diff);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            throw new NotImplementedException();
        }
    }
}
