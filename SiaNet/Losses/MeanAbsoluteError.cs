using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            return K.Reshape(K.Mean(K.Abs(preds - labels), 1), 1, -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            return (preds - labels) / ((float)preds.Shape[0] * K.Abs(preds - labels));
        }
    }
}
