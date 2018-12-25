using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Losses;
using TensorSharp.Expression;

namespace SiaNet.Metrics
{
    public class MSLE : BaseMetric
    {
        public MSLE()
            :base("msle")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return new MeanSquaredLogError().Call(preds, labels);
        }
    }
}
