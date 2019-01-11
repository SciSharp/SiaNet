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
            var first_log = Log(Clip(preds, EPSILON, float.MaxValue) + 1);
            var second_log = Log(Clip(labels, EPSILON, float.MaxValue) + 1);

            return Mean(Square(first_log - second_log), 1).Reshape(1, -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            float norm = 2f / preds.Shape[0];
            var first_log = Log(Clip(preds, EPSILON, float.MaxValue) + 1);
            var second_log = Log(Clip(labels, EPSILON, float.MaxValue) + 1);

            return  norm * (first_log - second_log) / (Clip(preds, EPSILON, float.MaxValue) + 1);
        }
    }
}
