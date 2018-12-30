using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Losses;
using TensorSharp;

namespace SiaNet.Metrics
{
    public class MALE : BaseMetric
    {
        public MALE()
            :base("male")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            return new MeanSquaredLogError().Call(preds, labels);
        }
    }
}
