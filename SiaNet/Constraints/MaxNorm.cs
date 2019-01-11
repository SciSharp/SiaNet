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

        public uint? Axis { get; set; }

        public MaxNorm(float maxValue, uint? axis = null)
        {
            MaxValue = maxValue;
            Axis = axis;
        }

        public override Tensor Call(Tensor w)
        {
            Tensor norms = null;
            if(!Axis.HasValue)
            {
                norms = Sqrt(Sum(Square(w)));
            }
            else
            {
                norms = Sqrt(Sum(Square(w), (int)Axis.Value));
            }

            var desired = Clip(norms, 0, MaxValue);
            return w * (desired / (EPSILON + norms));
        }
    }
}
