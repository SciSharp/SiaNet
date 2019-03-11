using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Constraints
{
    /// <summary>
    /// 
    /// </summary>
    public class MinMaxNorm : BaseConstraint
    {
        /// <summary>
        /// 
        /// </summary>
        public float MinValue { get; set; }

        /// <summary>
        /// 
        /// </summary>
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

        internal override Tensor Call(Tensor w)
        {
            Tensor norms = null;
            norms = K.Sqrt(K.Sum(K.Square(w), (int)Axis));


            var desired = Rate * K.Clip(norms, MinValue, MaxValue) + (1 - Rate) * norms;
            w = w * (desired / (K.Epsilon() + norms));
            return w;
        }
    }
}
