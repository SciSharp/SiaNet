using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Engine;
using SiaNet.Losses;

namespace SiaNet.Metrics
{
    public class MAE : BaseMetric
    {
        public MAE()
            :base("mae")
        {

        }

        public override Tensor Calc(Tensor preds, Tensor labels)
        {
            return new MeanAbsoluteError().Forward(preds, labels);
        }
    }
}
