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
            return Mean(Maximum(1 - labels * preds, 0));
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            throw new NotImplementedException();
        }
    }
}
