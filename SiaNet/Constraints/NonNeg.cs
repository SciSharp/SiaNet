using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
