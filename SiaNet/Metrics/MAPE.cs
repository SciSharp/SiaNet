using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Engine;
using SiaNet.Losses;

namespace SiaNet.Metrics
{
    public class MAPE : BaseMetric
    {
        public MAPE()
            :base("mape")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            return new MeanAbsolutePercentageError().Call(preds, labels);
        }
    }
}
