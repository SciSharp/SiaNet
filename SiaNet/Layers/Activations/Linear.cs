namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    /// <summary>
    /// Linear activation with f(x)=x
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Linear : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Linear"/> class.
        /// </summary>
        public Linear()
            : base("linear")
        {
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = x;
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
        }
    }
}
