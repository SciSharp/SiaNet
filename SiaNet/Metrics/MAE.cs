namespace SiaNet.Metrics
{
    using SiaNet.Engine;
    using SiaNet.Losses;

    /// <summary>
    /// Mean Absolute Error (MAE) is a quantity used to measure how close forecasts or predictions are to the eventual outcomes
    /// </summary>
    /// <seealso cref="SiaNet.Metrics.BaseMetric" />
    public class MAE : BaseMetric
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MAE"/> class.
        /// </summary>
        public MAE()
            :base("mae")
        {

        }

        /// <summary>
        /// Calculates the metric with predicted and true values.
        /// </summary>
        /// <param name="preds">The predicted value.</param>
        /// <param name="labels">The true value.</param>
        /// <returns></returns>
        public override Tensor Calc(Tensor preds, Tensor labels)
        {
            return new MeanAbsoluteError().Forward(preds, labels);
        }
    }
}
