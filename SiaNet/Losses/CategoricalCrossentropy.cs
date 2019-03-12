namespace SiaNet.Losses
{
    using SiaNet.Engine;

    /// <summary>
    /// Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. 
    /// Cross-entropy loss increases as the predicted probability diverges from the actual label.
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class CategoricalCrossentropy : BaseLoss
    {
        /// <summary>
        /// Gets or sets a value indicating whether [from logit].
        /// </summary>
        /// <value>
        ///   <c>true</c> if [from logit]; otherwise, <c>false</c>.
        /// </value>
        public bool FromLogit { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="CategoricalCrossentropy"/> class.
        /// </summary>
        /// <param name="fromLogit">if set to <c>true</c> [from logit].</param>
        public CategoricalCrossentropy(bool fromLogit = false)
            : base("categorical_crossentropy")
        {
            FromLogit = fromLogit;
        }

        /// <summary>
        /// Forwards the inputs and calculate the loss.
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            if (FromLogit)
                preds = K.Softmax(preds);
            else
                preds /= K.Sum(preds, -1);

            preds = K.Clip(preds, K.Epsilon(), 1 - K.Epsilon());
            return K.Sum(-1 * labels * K.Log(preds), -1);
        }

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            preds = K.Clip(preds, K.Epsilon(), 1 - K.Epsilon());
            return (preds - labels) / preds;
        }
    }
}
