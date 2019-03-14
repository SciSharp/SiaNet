namespace SiaNet.Layers
{
    using SiaNet.Engine;

    /// <summary>
    /// Permutes the dimensions of the input according to a given pattern. Useful for e.g.connecting RNNs and convnets together.
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Permute : BaseLayer
    {
        /// <summary>
        /// Permutation pattern, does not include the samples dimension. Indexing starts at 1. For instance, (2, 1) permutes the first and second dimension of the input.
        /// </summary>
        /// <value>
        /// The dims.
        /// </value>
        public int[] Dims { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Permute"/> class.
        /// </summary>
        /// <param name="dims">The dims.</param>
        public Permute(params int[] dims)
            : base("permute")
        {
            Dims = dims;
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = x.Transpose(Dims);
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
