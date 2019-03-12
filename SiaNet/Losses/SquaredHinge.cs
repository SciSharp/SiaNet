using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Losses
{
    /// <summary>
    /// Squared Hinge Loss function is a variant of Hinge Loss, it solves the problem in hinge loss that the derivative of hinge loss has a discontinuity at Pred * True = 1
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class SquaredHinge : BaseLoss
    {
        public SquaredHinge()
            : base("squared_hinge")
        {

        }

        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            var value = 1 - labels * preds;

            return K.Mean(K.Square(K.Maximum(value, 0)), -1);
        }

        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            float norm = 2f / preds.Shape[0];
            return -1 * norm * labels * K.Maximum((1 - labels * preds), 0);
        }
    }
}
