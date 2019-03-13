namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    /// <summary>
    /// Exponential activation function which returns simple exp(x)
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Exp : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Exp"/> class.
        /// </summary>
        public Exp()
            : base("exp")
        {
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = K.Exp(x);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * Output;
        }
    }
}
