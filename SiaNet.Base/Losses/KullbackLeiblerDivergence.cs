using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Losses
{
    public class KullbackLeiblerDivergence : BaseLoss
    {
        public KullbackLeiblerDivergence()
            : base("kullback_leibler_divergence")
        {
        }

        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            var y_true = K.Clip(labels, K.Epsilon(), 1);
            var y_pred = K.Clip(preds, K.Epsilon(), 1);

            return K.Sum(y_true * K.Log(y_true / y_pred), -1);
        }

        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            var y_true = K.Clip(labels, K.Epsilon(), 1);
            var y_pred = K.Clip(preds, K.Epsilon(), 1);

            return K.Maximum((-1 * (y_true / y_pred)), 0);
        }
    }
}
