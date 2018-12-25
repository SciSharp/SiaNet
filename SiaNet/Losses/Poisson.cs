using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class Poisson : BaseLoss
    {
        public Poisson()
            : base("poisson")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return (labels - labels.CMul((preds + float.Epsilon).Log())).MeanAll();
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
