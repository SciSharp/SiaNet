namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    /// <summary>
    /// Hyperbolic tangent activation. Tanh squashes a real-valued number to the range [-1, 1].
    /// It’s non-linear. But unlike Sigmoid, its output is zero-centered. Therefore, in practice the tanh non-linearity is always preferred to the sigmoid nonlinearity.
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Tanh : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Tanh"/> class.
        /// </summary>
        public Tanh()
            : base("tanh")
        {
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = K.Tanh(x);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * (1 - K.Square(Output));
        }
    }
}
