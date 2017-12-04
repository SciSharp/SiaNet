namespace SiaNet.Model.Layers
{
    using System.Dynamic;

    /// <summary>
    /// Reshapes an output to a certain shape.
    /// </summary>
    public class Reshape : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Reshape"/> class.
        /// </summary>
        internal Reshape()
        {
            base.Name = "Reshape";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Reshape"/> class.
        /// </summary>
        /// <param name="targetshape">The target shape of the output.</param>
        public Reshape(int[] targetshape)
            : this()
        {
            TargetShape = targetshape;
            Shape = null;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Reshape"/> class.
        /// </summary>
        /// <param name="targetshape">The target shape of the output.</param>
        /// <param name="shape">The shape of the input data.</param>
        public Reshape(int[] targetshape, int[] shape)
            : this(targetshape)
        {
            Shape = shape;
        }

        /// <summary>
        /// List of integers. Does not include the batch axis.
        /// </summary>
        /// <value>
        /// The target shape.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int[] TargetShape
        {
            get
            {
                return base.Params.TargetShape;
            }

            set
            {
                base.Params.TargetShape = value;
            }
        }

        /// <summary>
        /// Gets or sets the input shape of the data. Used when the this layer is the first in the stack.
        /// </summary>
        /// <value>
        /// The shape.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public int[] Shape
        {
            get
            {
                return base.Params.Shape;
            }

            set
            {
                base.Params.Shape = value;
            }
        }
    }
}
