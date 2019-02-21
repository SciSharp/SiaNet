using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            preds = K.Argmax(preds, 1);
            labels = K.Argmax(labels, 1);

            var r = K.EqualTo(preds, labels);

            return r;
        }
    }
}
