using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Losses
{
    public class LogCosh : BaseLoss
    {
        public LogCosh()
            : base("logcosh")
        {

        }

        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            return K.Mean(_logcosh(preds - labels), -1);
        }

        private Tensor _logcosh(Tensor x)
        {
            return x + K.Softplus(-2 * x) - (float)Math.Log(2);
        }

        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            return -1 * K.Tanh(labels - preds) / preds.Shape[0];
        }
    }
}
