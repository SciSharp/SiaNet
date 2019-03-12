namespace SiaNet.Losses
{
    using SiaNet.Engine;

    /// <summary>
    /// The mean squared error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors—that is, the average squared difference between the estimated values and what is estimated.
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class MeanSquaredError : BaseLoss
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MeanSquaredError"/> class.
        /// </summary>
        public MeanSquaredError()
            : base("mean_squared_error")
        {

        }

        /// <summary>
        /// Forwards the inputs and calculate the loss.
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            return K.Mean(K.Square(preds - labels), -1);
        }

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            float norm = 2f / (float)preds.Shape[0];
            return (preds - labels) * norm;
        }
    }
}
