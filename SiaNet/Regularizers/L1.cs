namespace SiaNet.Regularizers
{
    /// <summary>
    /// L1 regularization technique also called Lasso Regression
    /// <para>
    /// Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function.
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Regularizers.L1L2" />
    public class L1 : L1L2
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L1"/> class.
        /// </summary>
        /// <param name="l">The value for the regularizer.</param>
        public L1(float l = 0.01f)
            : base(l, 0)
        {
            Name = "L1";
        }
    }
}
