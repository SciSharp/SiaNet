using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Metrics
{
    public class Accuracy : BaseMetric
    {
        public Accuracy()
            :base("accuracy")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            var predData = preds.Argmax(1);
            var labelData = labels.Argmax(1);

            
            return predData;
        }
    }
}
