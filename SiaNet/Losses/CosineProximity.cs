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
            var y_true = L2Normalize(labels);
            var y_pred = L2Normalize(preds);
            return -1 * Sum(labels * preds);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            throw new NotImplementedException();
        }
    }
}
