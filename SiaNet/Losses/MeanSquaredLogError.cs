using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            var first_log = K.Log(K.Clip(preds, K.Epsilon(), float.MaxValue) + 1);
            var second_log = K.Log(K.Clip(labels, K.Epsilon(), float.MaxValue) + 1);

            return K.Mean(K.Square(first_log - second_log), -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            float norm = 2f / preds.Shape[0];
            var first_log = K.Log(K.Clip(preds, K.Epsilon(), float.MaxValue) + 1);
            var second_log = K.Log(K.Clip(labels, K.Epsilon(), float.MaxValue) + 1);

            return  norm * (first_log - second_log) / (K.Clip(preds, K.Epsilon(), float.MaxValue) + 1);
        }
    }
}
