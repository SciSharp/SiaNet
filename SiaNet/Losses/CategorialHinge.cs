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
            var pos = Sum(labels * preds);
            var neg = Max((1 - labels) * labels);

            return Maximum(0f, neg - pos + 1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            throw new NotImplementedException();
        }
    }
}
