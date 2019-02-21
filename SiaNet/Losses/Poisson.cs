using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            return K.Mean(preds - labels * K.Log(preds + K.Epsilon()), -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            return (1 - (labels / (preds + K.Epsilon()))) / preds.Shape[0];
        }
    }
}
