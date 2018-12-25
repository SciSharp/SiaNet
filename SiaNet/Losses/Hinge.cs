using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class Hinge : BaseLoss
    {
        public Hinge()
            : base("hinge")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return (1 - labels.CMul(preds)).Max(0).MeanAll();
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
