using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Constraints
{
    /// <summary>
    /// 
    /// </summary>
    public class NonNeg : BaseConstraint
    {
        /// <summary>
        /// 
        /// </summary>
        public NonNeg()
        {
        }

        internal override Tensor Call(Tensor w)
        {
            w = w * (w >= 0);
            return w;
        }
    }
}
