namespace SiaNet.Initializers
{
    using SiaNet.Engine;

    /// <summary>
    /// Initializer that generates tensors initialized to a constant value.
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.BaseInitializer" />
    public class Constant : BaseInitializer
    {
        /// <summary>
        /// float; the value of the generator tensors.
        /// </summary>
        /// <value>
        /// The value.
        /// </value>
        public float Value { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Constant"/> class.
        /// </summary>
        /// <param name="value">float; the value of the generator tensors.</param>
        public Constant(float value)
            :base("constant")
        {
            Value = value;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Constant"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="value">The value.</param>
        internal Constant(string name, float value)
           : base(name)
        {
            Value = value;
        }

        /// <summary>
        /// Generates a tensor with specified shape.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns></returns>
        public override Tensor Generate(params long[] shape)
        {
            Tensor tensor = K.Constant(Value, shape);
            return tensor;
        }
    }
}
