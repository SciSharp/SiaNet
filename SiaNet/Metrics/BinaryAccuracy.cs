using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Metrics
{
    public class BinaryAccuracy : BaseMetric
    {
        public BinaryAccuracy()
            :base("binary_accuracy")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            var r = EqualTo(preds, Round(labels));

            return r;
        }
    }
}
