namespace SiaNet.Regularizers
{
    /// <summary>
    /// L2 regularization  technique also called Ridge Regression.
    /// <para>
    /// Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function. Here the highlighted part represents L2 regularization element.
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Regularizers.L1L2" />
    public class L2 : L1L2
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L2"/> class.
        /// </summary>
        /// <param name="l">The value for the regularizer.</param>
        public L2(float l = 0.01f)
            : base(0, l)
        {
            Name = "L2";
        }
    }
}
