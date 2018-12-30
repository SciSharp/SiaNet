using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

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
            
            return Mean(Square(Maximum(value, 0)));
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            throw new NotImplementedException();
        }
    }
}
