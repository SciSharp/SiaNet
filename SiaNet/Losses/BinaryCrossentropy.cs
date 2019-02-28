using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaNet.Engine;

namespace SiaNet.Losses
{
    public class BinaryCrossentropy : BaseLoss
    {
        public bool FromLogit { get; set; }

        public BinaryCrossentropy(bool fromLogit = false)
            : base("binary_crossentropy")
        {
            FromLogit = fromLogit;
        }

        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            Tensor output = preds;
            if (!FromLogit)
            {
                output = K.Clip(output, K.Epsilon(), 1f - K.Epsilon());
                output = K.Log(output / (1 - output));
            }

            float scale = (2f * preds.ElementCount) / 3f;
            output = K.Sigmoid(output);

            return K.Mean(labels * K.Neg(K.Log(output)) + (1 - labels) * K.Neg(K.Log(1 - output)), -1);
        }

        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            Tensor output = K.Clip(preds, K.Epsilon(), 1f - K.Epsilon());
            return K.Neg((labels - 1) / (1 - output) - labels / output);
        }
    }
}
