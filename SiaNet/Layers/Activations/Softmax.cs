namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    /// <summary>
    /// Softmax function calculates the probabilities distribution of the event over ‘n’ different events. 
    /// In general way of saying, this function will calculate the probabilities of each target class over all possible target classes. 
    /// Later the calculated probabilities will be helpful for determining the target class for the given inputs.
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Softmax : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Softmax"/> class.
        /// </summary>
        public Softmax()
            : base("softmax")
        {
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = Global.ActFunc.SoftmaxForward(x);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Global.ActFunc.SoftmaxBackward(Input.Data, outputgrad);
        }
    }
}
