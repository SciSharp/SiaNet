using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaDNN.Constraints
{
    public class MinMaxNorm : BaseConstraint
    {
        public float MinValue { get; set; }
        public float MaxValue { get; set; }
        public float Rate { get; set; }
        public uint Axis { get; set; }

        public MinMaxNorm(float minVale = 0, float maxValue = 1, float rate = 1f, uint axis = 0)
        {
            MinValue = minVale;
            MaxValue = maxValue;
            Rate = rate;
            Axis = axis;
        }

        public override NDArray Call(NDArray w)
        {
            var norms = NDArray.Sqrt(NDArray.Sum(NDArray.Square(w), new Shape(Axis), true));
            var desired = (NDArray.Clip(norms, MinValue, MaxValue) * Rate) + (norms * (1 - Rate));
            w *= desired / (norms + float.Epsilon);
            return w;
        }
    }
}
