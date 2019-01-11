using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class CategorialHinge : BaseLoss
    {
        public CategorialHinge()
            : base("categorical_hinge")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            var pos = Sum(labels * preds, -1);
            var neg = Max((1 - labels) * preds, -1);

            return Maximum(neg - pos + 1, 0f);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            var diff = (1 - labels) * preds - Sum(labels * preds, -1);
            return Maximum(diff, 0);
        }
    }
}
