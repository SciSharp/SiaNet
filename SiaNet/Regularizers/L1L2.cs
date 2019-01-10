using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Regularizers
{
    public class L1L2 : BaseRegularizer
    {
        public L1L2(float l1 = 0.01f, float l2 = 0.01f)
            : base(l1, l2)
        {

        }

        public override Tensor Call(Tensor x)
        {
            Tensor regularizer = new Tensor(Global.Device, DType.Float32, x.Shape);
            if (L1 > 0)
            {
                regularizer += Sum(L1 * Abs(x));
            }

            if (L2 > 0)
            {
                regularizer += Sum(L2 * Square(x));
            }

            return regularizer;
        }

        public override Tensor CalcGrad(Tensor x, Tensor grad)
        {
            if (L1 > 0)
            {
                grad = grad + (L1 * x);
            }

            if(L2 > 0)
            {
                grad = grad + (L1 * x) / (Abs(x) + float.Epsilon);
            }

            return grad;
        }
    }
}
