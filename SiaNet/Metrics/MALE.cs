using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Losses;
using TensorSharp.Expression;

namespace SiaNet.Metrics
{
    public class MALE : BaseMetric
    {
        public MALE()
            :base("male")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return new MeanSquaredLogError().Call(preds, labels);
        }
    }
}
