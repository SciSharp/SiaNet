using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Losses;
using TensorSharp.Expression;

namespace SiaNet.Metrics
{
    public class MSE : BaseMetric
    {
        public MSE()
            :base("mse")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return new MeanSquaredError().Call(preds, labels);
        }
    }
}
