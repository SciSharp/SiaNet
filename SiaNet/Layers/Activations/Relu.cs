namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    /// <summary>
    /// Rectified Linear Unit.
    ///<para>
    ///With default values, it returns element-wise max(x, 0).
    ///</para>
    ///<para>
    ///<![CDATA[Otherwise, it follows: f(x) = max_value for x >= max_value, f(x) = x for threshold <= x<max_value, f(x) = alpha* (x - threshold) otherwise.]]>
    ///</para>
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Relu : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Relu"/> class.
        /// </summary>
        public Relu()
            : base("relu")
        {
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = Global.ActFunc.ReluForward(x);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Global.ActFunc.ReluBackward(Input.Data, outputgrad);
        }
    }
}
