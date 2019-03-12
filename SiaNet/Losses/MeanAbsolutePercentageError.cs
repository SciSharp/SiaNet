namespace SiaNet.Losses
{
    using SiaNet.Engine;

    /// <summary>
    /// The mean absolute percentage error (MAPE), also known as mean absolute percentage deviation (MAPD), 
    /// is a measure of prediction accuracy of a forecasting method in statistics, for example in trend estimation, also used as a Loss function for regression problems in Machine Learning.
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class MeanAbsolutePercentageError : BaseLoss
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MeanAbsolutePercentageError"/> class.
        /// </summary>
        public MeanAbsolutePercentageError()
            : base("mean_absolute_percentage_error")
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
            var diff = K.Abs(preds - labels) / K.Clip(K.Abs(labels), K.Epsilon(), float.MaxValue);
            return 100 * K.Reshape(K.Mean(diff, 1), 1, -1);
        }

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            var diff = (preds - labels) / K.Clip(K.Abs(labels) * K.Abs(labels - preds), K.Epsilon(), float.MaxValue);
            return 100 * diff / preds.Shape[0];
        }
    }
}
