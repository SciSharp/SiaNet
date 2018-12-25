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

        public int Axis { get; set; }

        public MinMaxNorm(float minVale = 0, float maxValue = 1, float rate = 1f, int axis = 0)
        {
            MinValue = minVale;
            MaxValue = maxValue;
            Rate = rate;
            Axis = axis;
        }

        public override Tensor Call(Tensor w)
        {
            var norms = w.TVar().Pow(2).Sum(Axis).Sqrt();
            var desired = norms.Clamp(MinValue, MaxValue) + (norms * (1 - Rate));
            var wVar = desired.CDiv(norms + float.Epsilon);
            return wVar.Evaluate();
        }
    }
}
