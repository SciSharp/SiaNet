using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Constraints
{
    public class UnitNorm : BaseConstraint
    {
        public int Axis;

        public UnitNorm(int axis = 0)
        {
            Axis = axis;
        }

        public override Tensor Call(Tensor w)
        {
            return w / (K.Epsilon() + K.Sqrt(K.Sum(K.Square(w), Axis)));
        }
    }
}
