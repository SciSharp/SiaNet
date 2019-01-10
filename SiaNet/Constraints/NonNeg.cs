using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Constraints
{
    public class NonNeg : BaseConstraint
    {
        public NonNeg()
        {
        }

        public override Tensor Call(Tensor w)
        {
            w = w * (w >= 0);
            return w;
        }
    }
}
