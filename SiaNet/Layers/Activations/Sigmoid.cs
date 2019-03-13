namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    /// <summary>
    /// Sigmoid takes a real value as input and outputs another value between 0 and 1. 
    /// It’s easy to work with and has all the nice properties of activation functions: it’s non-linear, continuously differentiable, monotonic, and has a fixed output range.
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Sigmoid : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Sigmoid"/> class.
        /// </summary>
        public Sigmoid()
            : base("sigmoid")
        {
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = K.Sigmoid(x);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * Output * (1 - Output);
        }
    }
}
