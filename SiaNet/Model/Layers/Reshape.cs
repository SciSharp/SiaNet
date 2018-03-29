using Newtonsoft.Json;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Reshapes an output to a certain shape.
    /// </summary>
    public class Reshape : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Reshape" /> class.
        /// </summary>
        /// <param name="targetshape">The target shape of the output.</param>
        public Reshape(int[] targetshape)
            : this()
        {
            TargetShape = targetshape;
            Shape = null;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Reshape" /> class.
        /// </summary>
        /// <param name="targetshape">The target shape of the output.</param>
        /// <param name="shape">The shape of the input data.</param>
        public Reshape(int[] targetshape, int[] shape)
            : this(targetshape)
        {
            Shape = shape;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Reshape" /> class.
        /// </summary>
        internal Reshape()
        {
            Name = "Reshape";
        }

        /// <summary>
        ///     Gets or sets the input shape of the data. Used when the this layer is the first in the stack.
        /// </summary>
        /// <value>
        ///     The shape.
        /// </value>
        [JsonIgnore]
        public int[] Shape
        {
            get => GetParam<int[]>("Shape");

            set => SetParam("Shape", value);
        }

        /// <summary>
        ///     List of integers. Does not include the batch axis.
        /// </summary>
        /// <value>
        ///     The target shape.
        /// </value>
        [JsonIgnore]
        public int[] TargetShape
        {
            get => GetParam<int[]>("TargetShape");

            set => SetParam("TargetShape", value);
        }
    }
}