using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            var pos = K.Sum(labels * preds, -1);
            var neg = K.Max((1 - labels) * preds, -1);

            return K.Maximum(neg - pos + 1, 0f);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            var diff = (1 - labels) * preds - K.Sum(labels * preds, -1);
            return K.Maximum(diff, 0);
        }
    }
}
