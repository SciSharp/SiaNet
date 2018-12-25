using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Constraints
{
    public class MaxNorm : BaseConstraint
    {
        public float MaxValue { get; set; }

        public int Axis { get; set; }

        public MaxNorm(float maxValue, int axis = -1)
        {
            MaxValue = maxValue;
            Axis = axis;
        }

        public override Tensor Call(Tensor w)
        {
            var norms = w.TVar().Pow(2).Sqrt();
            var desired = norms.Clamp(0, MaxValue);

            var wVar = w.TVar().CMul((desired.CDiv(float.Epsilon + norms)));
            return wVar.Evaluate();
        }
    }
}
