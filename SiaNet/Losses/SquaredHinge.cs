using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Losses
{
    public class SquaredHinge : BaseLoss
    {
        public SquaredHinge()
            : base("squared_hinge")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            var value = 1 - labels * preds;

            return K.Mean(K.Square(K.Maximum(value, 0)), -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            float norm = 2f / preds.Shape[0];
            return -1 * norm * labels * K.Maximum((1 - labels * preds), 0);
        }
    }
}
