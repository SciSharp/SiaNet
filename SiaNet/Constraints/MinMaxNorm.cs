using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Constraints
{
    public class MinMaxNorm : BaseConstraint
    {
        public float MinValue { get; set; }

        public float MaxValue { get; set; }

        public float Rate { get; set; }

        public uint? Axis { get; set; }

        public MinMaxNorm(float minVale = 0, float maxValue = 1, float rate = 1f, uint? axis = 0)
        {
            MinValue = minVale;
            MaxValue = maxValue;
            Rate = rate;
            Axis = axis;
        }

        public override Tensor Call(Tensor w)
        {
            Tensor norms = null;
            if (!Axis.HasValue)
                norms = Sqrt(Sum(Square(w)));
            else
                norms = Sqrt(Sum(Square(w), (int)Axis.Value));

            var desired = Rate * Clip(norms, MinValue, MaxValue) + (1 - Rate) * norms;
            w = w * (desired / (EPSILON + norms));
            return w;
        }
    }
}
