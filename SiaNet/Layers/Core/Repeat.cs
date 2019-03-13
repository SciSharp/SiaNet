namespace SiaNet.Layers
{
    using SiaNet.Engine;

    /// <summary>
    /// Repeat the input for specified number of times, for an axis
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Repeat : BaseLayer
    {
        /// <summary>
        /// Axis to which the tensor will be repeated
        /// </summary>
        /// <value>
        /// The axis.
        /// </value>
        public int Axis { get; set; }

        /// <summary>
        /// Integer, repetition factor
        /// </summary>
        /// <value>
        /// The number times.
        /// </value>
        public int NumTimes { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Repeat"/> class.
        /// </summary>
        /// <param name="numTimes">Integer, repetition factor.</param>
        /// <param name="axis">Axis to which the tensor will be repeated</param>
        public Repeat(int numTimes, int axis = 0)
            : base("repeatvector")
        {
            NumTimes = numTimes;
            Axis = axis;
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = K.Tile(x, NumTimes, Axis);
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
