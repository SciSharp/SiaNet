namespace SiaNet.Losses
{
    using SiaNet.Engine;

    /// <summary>
    /// Poisson loss function is a measure of how the predicted distribution diverges from the expected distribution, 
    /// the poisson as loss function is a variant from Poisson Distribution, where the poisson distribution is widely used for modeling count data. 
    /// It can be shown to be the limiting distribution for a normal approximation to a binomial where the number of trials goes to infinity and the probability
    /// goes to zero and both happen at such a rate that np is equal to some mean frequency for the process. 
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class Poisson : BaseLoss
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Poisson"/> class.
        /// </summary>
        public Poisson()
            : base("poisson")
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
            return K.Mean(preds - labels * K.Log(preds + K.Epsilon()), -1);
        }

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            return (1 - (labels / (preds + K.Epsilon()))) / preds.Shape[0];
        }
    }
}
