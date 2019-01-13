using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public class CosineProximity : BaseLoss
    {
        public CosineProximity()
            : base("cosine_proximity")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            return -1 * _cossine_sim(preds, labels);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            var y_true = Max(Sum(labels, -1), -1) / (Abs(preds * Abs(labels)));
            var y_pred = Max(Sum(preds, -1), -1) / Square(Abs(preds));

            return y_true + _cossine_sim(preds, labels) * y_pred;
        }

        private Tensor _cossine_sim(Tensor preds, Tensor labels)
        {
            var y_true = L2Normalize(labels, -1);
            var y_pred = L2Normalize(preds, -1);
            return Sum(y_true * y_pred, -1);
        }
    }
}
