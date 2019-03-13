using SiaNet.Engine;

namespace SiaNet.Layers.Activations
{
    /// <summary>
    /// Leaky version of a Rectified Linear Unit.
    /// <para>
    /// <![CDATA[It allows a small gradient when the unit is not active: f(x) = alpha* x for x< 0, f(x) = x for x >= 0.]]>
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class LeakyRelu : BaseLayer
    {
        /// <summary>
        /// Negative slope coefficient..
        /// </summary>
        /// <value>
        /// The alpha.
        /// </value>
        public float Alpha { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="LeakyRelu"/> class.
        /// </summary>
        /// <param name="alpha">Negative slope coefficient.</param>
        public LeakyRelu(float alpha = 0.3f)
            : base("leaky_relu")
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
            Output = Global.ActFunc.LeakyReluForward(Alpha, x);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Global.ActFunc.LeakyReluBackward(Alpha, Input.Data, outputgrad);
        }
    }
}
