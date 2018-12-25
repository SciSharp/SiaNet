using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

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
            return w.TVar().CDiv(w.TVar().Sum(Axis).Sqrt()).Evaluate();
        }
    }
}
