using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Engine;
using SiaNet.Losses;

namespace SiaNet.Metrics
{
    public class MSE : BaseMetric
    {
        public MSE()
            :base("mse")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            return new MeanSquaredError().Call(preds, labels);
        }
    }
}
