namespace SiaNet.Metrics
{
    using SiaNet.Engine;

    /// <summary>
    /// Accuracy is suggested to use to measure how accurate is the overall performance of a model is, considering both positive and negative classes without worrying about the imbalance of a data set
    /// </summary>
    /// <seealso cref="SiaNet.Metrics.BaseMetric" />
    public class Accuracy : BaseMetric
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Accuracy"/> class.
        /// </summary>
        public Accuracy()
            :base("accuracy")
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
            preds = K.Argmax(preds, 1);
            labels = K.Argmax(labels, 1);

            var r = K.EqualTo(preds, labels);

            return r;
        }
    }
}
