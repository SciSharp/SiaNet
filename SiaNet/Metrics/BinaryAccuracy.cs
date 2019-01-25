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
            preds = Clip(preds, 0, 1);
            var r = EqualTo(Round(preds), labels);

            return r;
        }
    }
}
