namespace SiaNet.Losses
{
    using SiaNet.Engine;

    /// <summary>
    /// Hinge Loss, also known as max-margin objective, is a loss function used for training classifiers. 
    /// The hinge loss is used for “maximum-margin” classification, most notably for support vector machines (SVMs).
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class Hinge : BaseLoss
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Hinge"/> class.
        /// </summary>
        public Hinge()
            : base("hinge")
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
            return K.Mean(K.Maximum(1 - labels * preds, 0), -1);
        }

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            return K.Neg(K.Maximum(labels / preds.Shape[0], 0));
        }
    }
}
