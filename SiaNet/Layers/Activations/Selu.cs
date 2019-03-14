namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    /// <summary>
    /// SELU is equal to: scale * elu(x, alpha), where alpha and scale are predefined constants. 
    /// The values of alpha and scale are chosen so that the mean and variance of the inputs are preserved between two consecutive layers as long as the weights are initialized correctly (see lecun_normal initialization) 
    /// and the number of inputs is "large enough"
    /// </summary>
    /// <seealso cref="SiaNet.Layers.Activations.Elu" />
    public class Selu : Elu
    {
        private static float alpha = 1.6732632423543772848170429916717f;
        private static float scale = 1.0507009873554804934193349852946f;

        /// <summary>
        /// Initializes a new instance of the <see cref="Selu"/> class.
        /// </summary>
        public Selu()
            : base(alpha)
        {
            Name = "selu";
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
           
            base.Forward(x);
            Output = scale * Output;
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            base.Backward(outputgrad);
            Input.Grad = scale * Input.Grad;
        }
    }
}
