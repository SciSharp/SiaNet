using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Losses;
using TensorSharp.Expression;

namespace SiaNet.Metrics
{
    public class MAPE : BaseMetric
    {
        public MAPE()
            :base("mape")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return new MeanAbsolutePercentageError().Call(preds, labels);
        }
    }
}
