using CNTK;

namespace SiaNet.Initializers
{
    /// <summary>
    ///     Initializer that generates tensors initialized to a constant value.
    /// </summary>
    /// <seealso cref="InitializerBase" />
    public class Constant : InitializerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Constant" /> class.
        /// </summary>
        /// <param name="value">The value of the generator tensors.</param>
        public Constant(double value)
        {
            Value = value;
        }

        public double Value { get; protected set; }

        /// <inheritdoc />
        internal override CNTKDictionary ToDictionary()
        {
            return CNTKLib.ConstantInitializer(Value);
        }
    }
}