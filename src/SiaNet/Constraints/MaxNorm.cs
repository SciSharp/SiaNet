using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaDNN.Constraints
{
    public class MaxNorm : BaseConstraint
    {
        public float MaxValue { get; set; }
        public int Axis { get; set; }

        public MaxNorm(float maxValue, int axis=-1)
        {
            MaxValue = maxValue;
            Axis = axis;
        }

        public override NDArray Call(NDArray w)
        {
            var norms = NDArray.Sqrt(NDArray.Square(w));
            var desired = NDArray.Clip(norms, 0, MaxValue);
            w *= (desired / (norms + float.Epsilon));
            return w;
        }
    }
}
