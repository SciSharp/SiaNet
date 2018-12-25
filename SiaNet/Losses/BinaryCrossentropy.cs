using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

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

        public override TVar Call(TVar preds, TVar labels)
        {
            TVar output = null;
            if(!FromLogit)
            {
                output = labels.Clamp(1e-7f, 1f - 1e-7f);
                output = output.CDiv(1 - output).Log();
            }

            return (preds.CMul(-output.Sigmoid().Log()) + (1 - preds).CMul(-(1 - output.Sigmoid()).Log()));
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
