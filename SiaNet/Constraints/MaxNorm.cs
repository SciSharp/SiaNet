using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Constraints
{
    /// <summary>
    /// 
    /// </summary>
    public class MaxNorm : BaseConstraint
    {
        /// <summary>
        /// 
        /// </summary>
        public float MaxValue { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public int Axis { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="maxValue"></param>
        /// <param name="axis"></param>
        public MaxNorm(float maxValue = 2, int axis = 0)
        {
            MaxValue = maxValue;
            Axis = axis;
        }

        internal override Tensor Call(Tensor w)
        {
            Tensor norms = K.Sqrt(K.Sum(K.Square(w), Axis));

            var desired = K.Clip(norms, 0, MaxValue);
            return w * (desired / (K.Epsilon() + norms));
        }
    }
}
