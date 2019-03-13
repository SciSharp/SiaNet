namespace SiaNet.Layers.Activations
{
    using SiaNet.Constraints;
    using SiaNet.Initializers;
    using SiaNet.Regularizers;
    using System.Linq;
    using SiaNet.Engine;

    /// <summary>
    /// Parametric Rectified Linear Unit.
    /// <para>
    /// <![CDATA[It follows: f(x) = alpha* x for x< 0, f(x) = x for x >= 0, where alpha is a learned array with the same shape as x.]]>
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class PRelu : BaseLayer
    {
        /// <summary>
        /// Gets or sets the initializer for the alpha parameter.
        /// </summary>
        /// <value>
        /// The alpha initializer.
        /// </value>
        public BaseInitializer AlphaInitializer { get; set; }

        /// <summary>
        /// Gets or sets the regularizer function for alpha parameter.
        /// </summary>
        /// <value>
        /// The alpha regularizer.
        /// </value>
        public BaseRegularizer AlphaRegularizer { get; set; }

        /// <summary>
        /// Gets or sets the constraint function for the alpha parameter.
        /// </summary>
        /// <value>
        /// The alpha constraint.
        /// </value>
        public BaseConstraint AlphaConstraint { get; set; }

        /// <summary>
        /// The position relu
        /// </summary>
        private Relu pos_relu;

        /// <summary>
        /// The neg relu
        /// </summary>
        private Relu neg_relu;

        /// <summary>
        /// Initializes a new instance of the <see cref="PRelu"/> class.
        /// </summary>
        /// <param name="alphaInitializer">The alpha initializer.</param>
        /// <param name="alphaRegularizer">The alpha regularizer.</param>
        /// <param name="alphaConstraint">The alpha constraint.</param>
        /// <param name="sharedAxes">The shared axes.</param>
        public PRelu(BaseInitializer alphaInitializer = null, BaseRegularizer alphaRegularizer=null, BaseConstraint alphaConstraint = null, params long[] sharedAxes)
            : base("prelu")
        {
            AlphaInitializer = alphaInitializer ?? new Zeros();
            AlphaRegularizer = alphaRegularizer;
            AlphaConstraint = alphaConstraint;
            pos_relu = new Relu();
            neg_relu = new Relu();
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            //ToDo: Implement shared axes
            base.Forward(x);
            long[] paramShape = x.Shape.ToList().Skip(1).ToArray();

            Parameter alpha = BuildParam("a", paramShape, x.ElementType, AlphaInitializer, AlphaConstraint, AlphaRegularizer);
            pos_relu.Forward(x);
            var pos = pos_relu.Output;

            neg_relu.Forward(-1 * x);
            var neg = -1f * alpha.Data * neg_relu.Output;
            Output = pos + neg;
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            pos_relu.Backward(outputgrad);
            neg_relu.Backward(outputgrad);

            Input.Grad = pos_relu.Input.Grad - Params["a"].Data * neg_relu.Input.Grad;
        }
    }
}
