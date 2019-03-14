namespace SiaNet.Losses
{
    using SiaNet.Engine;

    /// <summary>
    /// KL Divergence, also known as relative entropy, information divergence/gain, is a measure of how one probability distribution diverges from a second expected probability distribution.
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class KullbackLeiblerDivergence : BaseLoss
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="KullbackLeiblerDivergence"/> class.
        /// </summary>
        public KullbackLeiblerDivergence()
            : base("kullback_leibler_divergence")
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
            var y_true = K.Clip(labels, K.Epsilon(), 1);
            var y_pred = K.Clip(preds, K.Epsilon(), 1);

            return K.Sum(y_true * K.Log(y_true / y_pred), -1);
        }

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            var y_true = K.Clip(labels, K.Epsilon(), 1);
            var y_pred = K.Clip(preds, K.Epsilon(), 1);

            return K.Maximum((-1 * (y_true / y_pred)), 0);
        }
    }
}
