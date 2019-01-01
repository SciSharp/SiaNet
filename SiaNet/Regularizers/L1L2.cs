using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Regularizers
{
    public class L1L2 : BaseRegularizer
    {
        public L1L2(float l1 = 0.01f, float l2 = 0.01f)
            : base(l1, l2)
        {

        }

        public override Tensor Call(TVar x)
        {
            Tensor regularizer = new Tensor(Global.Device, DType.Float32, x.Evaluate().Shape);
            if (L1 > 0)
            {
                regularizer = (regularizer.TVar() + (x.Abs() * L1).SumAll()).Evaluate();
            }

            if (L2 > 0)
            {
                regularizer = (regularizer.TVar() + (x.Pow(2) * L2).SumAll()).Evaluate();
            }

            return regularizer;
        }

        public override Tensor CalcGrad(TVar x, TVar grad)
        {
            if (L1 > 0)
            {
                grad = grad + (L1 * x);
            }

            if(L2 > 0)
            {
                grad = grad + (L1 * x).CDiv((x.Abs() + float.Epsilon));
            }

            return grad.Evaluate();
        }
    }
}
