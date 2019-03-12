namespace SiaNet.Losses
{
    using SiaNet.Engine;
    using System;

    /// <summary>
    /// Logarithm of the hyperbolic cosine of the prediction error.
    ///<para>
    ///log(cosh(x)) is approximately equal to(x** 2) / 2 for small x and to abs(x) - log(2) for large x.
    ///This means that 'logcosh' works mostly like the mean squared error, but will not be so strongly affected by the occasional wildly incorrect prediction.
    ///</para>
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class LogCosh : BaseLoss
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LogCosh"/> class.
        /// </summary>
        public LogCosh()
            : base("logcosh")
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
            return K.Mean(_logcosh(preds - labels), -1);
        }

        /// <summary>
        /// Logcosh the specified x.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        private Tensor _logcosh(Tensor x)
        {
            return x + K.Softplus(-2 * x) - (float)Math.Log(2);
        }

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            return -1 * K.Tanh(labels - preds) / preds.Shape[0];
        }
    }
}
