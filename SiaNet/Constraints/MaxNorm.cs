using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Constraints
{
    public class MaxNorm : BaseConstraint
    {
        public float MaxValue { get; set; }

        public int Axis { get; set; }

        public MaxNorm(float maxValue = 2, int axis = 0)
        {
            MaxValue = maxValue;
            Axis = axis;
        }

        public override Tensor Call(Tensor w)
        {
            Tensor norms = K.Sqrt(K.Sum(K.Square(w), Axis));

            var desired = K.Clip(norms, 0, MaxValue);
            return w * (desired / (K.Epsilon() + norms));
        }
    }
}
