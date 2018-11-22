using CNTK;

namespace SiaNet.Initializers
{
    /// <summary>
    ///     Initializer that generates tensors initialized to 0.
    /// </summary>
    /// <seealso cref="InitializerBase" />
    public class Zeros : InitializerBase
    {
        /// <inheritdoc />
        internal override CNTKDictionary ToDictionary()
        {
            return CNTKLib.ConstantInitializer(0);
        }
    }
}