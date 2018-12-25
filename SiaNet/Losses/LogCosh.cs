using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class LogCosh : BaseLoss
    {
        public LogCosh()
            : base("logcosh")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            return (preds - labels).MeanAll();
        }

        private TVar _logcosh(TVar x)
        {
            return (x + (-2 * x).Softplus() - TVar.Fill(2, Global.Device, DType.Float32, 1).Log());
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
