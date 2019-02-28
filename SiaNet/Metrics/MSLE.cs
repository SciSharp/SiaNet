using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Engine;
using SiaNet.Losses;

namespace SiaNet.Metrics
{
    public class MSLE : BaseMetric
    {
        public MSLE()
            :base("msle")
        {

        }

        public override Tensor Calc(Tensor preds, Tensor labels)
        {
            return new MeanSquaredLogError().Forward(preds, labels);
        }
    }
}
