using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Metrics
{
    public class BinaryAccuracy : BaseMetric
    {
        public BinaryAccuracy()
            :base("binary_accuracy")
        {

        }

        public override Tensor Calc(Tensor preds, Tensor labels)
        {
            preds = K.Clip(preds, 0, 1);
            var r = K.EqualTo(K.Round(preds), labels);

            return r;
        }
    }
}
