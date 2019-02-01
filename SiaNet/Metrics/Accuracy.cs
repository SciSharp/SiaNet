using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Metrics
{
    public class Accuracy : BaseMetric
    {
        public Accuracy()
            :base("accuracy")
        {

        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            preds = Argmax(preds, 1);
            labels = Argmax(labels, 1);

            var r = EqualTo(preds, labels);

            return r;
        }
    }
}
