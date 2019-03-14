namespace SiaNet.Layers
{
    using SiaNet.Engine;

    /// <summary>
    /// Applies Dropout to the input. Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Dropout : BaseLayer
    {
        /// <summary>
        /// The noise
        /// </summary>
        private Tensor noise;

        /// <summary>
        /// float between 0 and 1. Fraction of the input units to drop.
        /// </summary>
        /// <value>
        /// The rate.
        /// </value>
        public float Rate { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Dropout"/> class.
        /// </summary>
        /// <param name="rate">float between 0 and 1. Fraction of the input units to drop..</param>
        public Dropout(float rate)
            :base("dropout")
        {
            SkipPred = true;
            Rate = rate;
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            
            var p = 1 - Rate;

            if (noise == null)
            {
                noise = K.RandomBernoulli(x.Shape, p);
            }

            Output = noise * p;
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * noise;
        }
    }
}
