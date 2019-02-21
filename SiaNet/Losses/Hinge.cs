using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Losses
{
    public class Hinge : BaseLoss
    {
        public Hinge()
            : base("hinge")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            return K.Mean(K.Maximum(1 - labels * preds, 0), -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            return K.Neg(K.Maximum(labels / preds.Shape[0], 0));
        }
    }
}
