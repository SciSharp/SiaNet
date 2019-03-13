namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    /// <summary>
    /// The softplus activation: log(exp(x) + 1).
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Softplus : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Softplus"/> class.
        /// </summary>
        public Softplus()
            : base("softplus")
        {
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = K.Softplus(x);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * (K.Exp(Input.Data) / (K.Exp(Input.Data) + 1));
        }
    }
}
