using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaNet.Engine;

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

        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            if (FromLogit)
                preds = K.Softmax(preds);
            else
                preds /= K.Sum(preds, -1);

            preds = K.Clip(preds, K.Epsilon(), 1 - K.Epsilon());
            return K.Sum(-1 * labels * K.Log(preds), -1);
        }

        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            preds = K.Clip(preds, K.Epsilon(), 1 - K.Epsilon());
            return (preds - labels) / preds;
        }
    }
}
