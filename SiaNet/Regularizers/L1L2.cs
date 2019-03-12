namespace SiaNet.Regularizers
{
    using SiaNet.Engine;

    /// <summary>
    /// Combined regularizer for Lasso and Ridge regression technique
    /// </summary>
    /// <seealso cref="SiaNet.Regularizers.BaseRegularizer" />
    public class L1L2 : BaseRegularizer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L1L2"/> class.
        /// </summary>
        /// <param name="l1">The l1 value. Default to 0.01</param>
        /// <param name="l2">The l2 value. Default to 0.01</param>
        public L1L2(float l1 = 0.01f, float l2 = 0.01f)
            : base(l1, l2)
        {
            Name = "L1L2";
        }

        /// <summary>
        /// Calls the specified x.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        internal override float Call(Tensor x)
        {
            float result = 0;
            if (L1 > 0)
            {
                result += K.Sum(L1 * K.Abs(x));
            }

            if (L2 > 0)
            {
                result += K.Sum(L2 * K.Square(x));
            }

            return result;
        }

        /// <summary>
        /// Calculates the grad.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        internal override Tensor CalcGrad(Tensor x)
        {
            Tensor grad = null;

            if (L1 > 0)
            {
                grad = (L1 * x) / (K.Abs(x) + K.Epsilon()); 
            }

            if(L2 > 0)
            {
                grad = (2 * L2 * x);
            }

            return grad;
        }
    }
}
