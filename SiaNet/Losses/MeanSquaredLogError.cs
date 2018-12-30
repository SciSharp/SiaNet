using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public class MeanSquaredLogError : BaseLoss
    {
        public MeanSquaredLogError()
            : base("mean_squared_logarithmic_error")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            var first_log = Log(preds + 1);
            var second_log = Log(labels + 1);

            return Mean(Square(first_log - second_log));
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            throw new NotImplementedException();
        }
    }
}
