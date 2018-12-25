using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Losses
{
    public class CategorialHinge : BaseLoss
    {
        public CategorialHinge()
            : base("categorical_hinge")
        {

        }

        public override TVar Call(TVar preds, TVar labels)
        {
            var pos = labels.CMul(preds).SumAll();
            var neg = (1 - labels).CMul(labels).MaxAll();

            return TVar.Fill(Math.Max(0f, (neg.ToScalar().Evaluate() - pos.ToScalar().Evaluate() + 1)), Global.Device, DType.Float32, 1);
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            throw new NotImplementedException();
        }
    }
}
