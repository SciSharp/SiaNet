namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    /// <summary>
    /// <![CDATA[Exponential linear unit activation function: x if x > 0 and alpha * (exp(x)-1) if x < 0.]]>
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Elu : BaseLayer
    {
        /// <summary>
        /// Slope of the negative section
        /// </summary>
        /// <value>
        /// The alpha.
        /// </value>
        public float Alpha { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Elu"/> class.
        /// </summary>
        /// <param name="alpha">Slope of the negative section.</param>
        public Elu(float alpha = 1)
            : base("elu")
        {
            Alpha = alpha;
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = Global.ActFunc.EluForward(Alpha, x);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Global.ActFunc.EluBackward(Alpha, Input.Data, outputgrad);
        }
    }
}
