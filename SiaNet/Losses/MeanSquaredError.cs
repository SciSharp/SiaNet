using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            return K.Mean(K.Square(preds - labels), -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            float norm = 2f / (float)preds.Shape[0];
            return (preds - labels) * norm;
        }
    }
}
