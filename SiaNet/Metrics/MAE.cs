using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Losses;
using TensorSharp.Expression;

namespace SiaNet.Metrics
{
    public class MAE : BaseMetric
    {
        public MAE()
            :base("mae")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return new MeanAbsoluteError().Call(preds, labels);
        }
    }
}
