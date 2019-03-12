using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Constraints
{
    /// <summary>
    /// Constrains the weights to be non-negative.
    /// </summary>
    /// <seealso cref="SiaNet.Constraints.BaseConstraint" />
    public class NonNeg : BaseConstraint
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NonNeg"/> class.
        /// </summary>
        public NonNeg()
        {
        }

        /// <summary>
        /// Invoke the constraint
        /// </summary>
        /// <param name="w">The w.</param>
        /// <returns></returns>
        internal override Tensor Call(Tensor w)
        {
            w = w * (w >= 0);
            return w;
        }
    }
}
