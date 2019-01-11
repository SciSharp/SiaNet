using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Constraints
{
    public class UnitNorm : BaseConstraint
    {
        public uint? Axis;

        public UnitNorm(uint? axis = 0)
        {
            Axis = axis;
        }

        public override Tensor Call(Tensor w)
        {
            if(!Axis.HasValue)
                w = w / (EPSILON + Sqrt(Sum(Square(w))));
            else
                w = w / (EPSILON + Sqrt(Sum(Square(w), (int)Axis.Value)));

            return w;
        }
    }
}
