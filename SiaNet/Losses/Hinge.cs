using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

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
            return Mean(Maximum(1 - labels * preds, 0), -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            return -1 * Maximum(labels / preds.Shape[0], 0);
        }
    }
}
