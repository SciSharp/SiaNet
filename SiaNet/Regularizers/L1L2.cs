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
            Name = "L1L2";
        }

        public override float Call(Tensor x)
        {
            float result = 0;
            if (L1 > 0)
            {
                result += SumF(L1 * Abs(x));
            }

            if (L2 > 0)
            {
                result += SumF(L2 * Square(x));
            }

            return result;
        }

        public override Tensor CalcGrad(Tensor x)
        {
            Tensor grad = null;

            if (L1 > 0)
            {
                grad = (L1 * x) / (Abs(x) + EPSILON); 
            }

            if(L2 > 0)
            {
                grad = (2 * L2 * x);
            }

            return grad;
        }
    }
}
