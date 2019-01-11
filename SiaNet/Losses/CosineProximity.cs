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
            var y_true = L2Normalize(labels, -1);
            var y_pred = L2Normalize(preds, -1);
            y_true.Print();
            y_pred.Print();
            return -1 * Sum(y_true * y_pred, -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            var y_true = L2Normalize(labels, -1);
            return -1 * labels / preds.Shape[0];
        }
    }
}
