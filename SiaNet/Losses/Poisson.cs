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
            return Mean(preds - labels * Log(preds + float.Epsilon));
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            throw new NotImplementedException();
        }
    }
}
