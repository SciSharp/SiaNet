using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Losses
{
    public class CosineProximity : BaseLoss
    {
        public CosineProximity()
            : base("cosine_proximity")
        {

        }

        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            return -1 * _cossine_sim(preds, labels);
        }

        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            var y_true = K.Max(K.Sum(labels, -1), -1) / (K.Abs(preds * K.Abs(labels)));
            var y_pred = K.Max(K.Sum(preds, -1), -1) / K.Square(K.Abs(preds));

            return y_true + _cossine_sim(preds, labels) * y_pred;
        }

        private Tensor _cossine_sim(Tensor preds, Tensor labels)
        {
            var y_true = K.L2Normalize(labels, -1);
            var y_pred = K.L2Normalize(preds, -1);
            return K.Sum(y_true * y_pred, -1);
        }
    }
}
