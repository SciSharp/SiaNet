namespace SiaNet.Losses
{
    using SiaNet.Engine;

    /// <summary>
    /// Binary Cross-Entropy Loss. Also called Sigmoid Cross-Entropy loss. 
    /// It is a Sigmoid activation plus a Cross-Entropy loss. 
    /// Unlike Softmax loss it is independent for each vector component (class), meaning that the loss computed for every vector component is not affected by other component values
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class BinaryCrossentropy : BaseLoss
    {
        /// <summary>
        /// Gets or sets a value indicating whether [from logit].
        /// </summary>
        /// <value>
        ///   <c>true</c> if [from logit]; otherwise, <c>false</c>.
        /// </value>
        public bool FromLogit { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryCrossentropy"/> class.
        /// </summary>
        /// <param name="fromLogit">if set to <c>true</c> [from logit].</param>
        public BinaryCrossentropy(bool fromLogit = false)
            : base("binary_crossentropy")
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
            Tensor output = preds;
            if (!FromLogit)
            {
                output = K.Clip(output, K.Epsilon(), 1f - K.Epsilon());
                output = K.Log(output / (1 - output));
            }

            float scale = (2f * preds.ElementCount) / 3f;
            output = K.Sigmoid(output);

            return K.Mean(labels * K.Neg(K.Log(output)) + (1 - labels) * K.Neg(K.Log(1 - output)), -1);
        }

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            Tensor output = K.Clip(preds, K.Epsilon(), 1f - K.Epsilon());
            return K.Neg((labels - 1) / (1 - output) - labels / output);
        }
    }
}
