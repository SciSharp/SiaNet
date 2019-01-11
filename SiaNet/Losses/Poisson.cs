using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public class Poisson : BaseLoss
    {
        public Poisson()
            : base("poisson")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            return Mean(preds - labels * Log(preds + EPSILON), -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            return (1 - (labels / (preds + EPSILON))) / preds.Shape[0];
        }
    }
}
