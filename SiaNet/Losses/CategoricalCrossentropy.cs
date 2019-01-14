using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using TensorSharp;

namespace SiaNet.Losses
{
    public class CategoricalCrossentropy : BaseLoss
    {
        public bool FromLogit { get; set; }

        public CategoricalCrossentropy(bool fromLogit = false)
            : base("categorical_crossentropy")
        {
            FromLogit = fromLogit;
        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            if (FromLogit)
                preds = Softmax(preds);
            else
                preds /= Sum(preds, -1);

            preds = Clip(preds, EPSILON, 1 - EPSILON);
            return Sum(-1 * labels * Log(preds), -1);
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            preds = Clip(preds, EPSILON, 1 - EPSILON);
            return (preds - labels) / preds;
        }
    }
}
