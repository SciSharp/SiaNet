namespace SiaNet.Layers
{
    using SiaNet.Engine;

    /// <summary>
    /// Flattens the input. Does not affect the batch size.
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Flatten : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Flatten"/> class.
        /// </summary>
        public Flatten()
             : base("flatten")
        {

        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);

            Output = x.Reshape(x.Shape[0], -1);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad.Reshape(Input.Data.Shape);
        }
    }
}
