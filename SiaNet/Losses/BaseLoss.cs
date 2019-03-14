using SiaNet.Engine;

namespace SiaNet.Losses
{
    /// <summary>
    /// A loss function (or objective function, or optimization score function) is one of the two parameters required to compile a model.
    /// </summary>
    public abstract class BaseLoss
    {
        /// <summary>
        /// The backend instance
        /// </summary>
        internal IBackend K = Global.CurrentBackend;

        /// <summary>
        /// Gets or sets the name of the loss function
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseLoss"/> class.
        /// </summary>
        /// <param name="name">The name of the loss function.</param>
        public BaseLoss(string name)
        {
            Name = name;
        }

        /// <summary>
        /// Forwards the inputs and calculate the loss.
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public abstract Tensor Forward(Tensor preds, Tensor labels);

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public abstract Tensor Backward(Tensor preds, Tensor labels);

        /// <summary>
        /// Gets the specified loss type.
        /// </summary>
        /// <param name="lossType">Type of the loss.</param>
        /// <returns></returns>
        internal static BaseLoss Get(LossType lossType)
        {
            BaseLoss loss = null;
            switch (lossType)
            {
                case LossType.MeanSquaredError:
                    loss = new MeanSquaredError();
                    break;
                case LossType.MeanAbsoluteError:
                    loss = new MeanAbsoluteError();
                    break;
                case LossType.MeanAbsolutePercentageError:
                    loss = new MeanAbsolutePercentageError();
                    break;
                case LossType.MeanAbsoluteLogError:
                    loss = new MeanSquaredLogError();
                    break;
                case LossType.SquaredHinge:
                    loss = new SquaredHinge();
                    break;
                case LossType.Hinge:
                    loss = new Hinge();
                    break;
                case LossType.BinaryCrossEntropy:
                    loss = new BinaryCrossentropy();
                    break;
                case LossType.CategorialCrossEntropy:
                    loss = new CategoricalCrossentropy();
                    break;
                case LossType.CTC:
                    break;
                case LossType.KullbackLeiblerDivergence:
                    loss = new KullbackLeiblerDivergence();
                    break;
                case LossType.Logcosh:
                    loss = new LogCosh();
                    break;
                case LossType.Poisson:
                    loss = new Poisson();
                    break;
                case LossType.CosineProximity:
                    loss = new CosineProximity();
                    break;
                default:
                    break;
            }

            return loss;
        }
    }
}
