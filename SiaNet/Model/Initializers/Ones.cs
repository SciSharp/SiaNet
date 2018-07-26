using CNTK;

namespace SiaNet.Model.Initializers
{
    /// <summary>
    ///     Initializer that generates tensors initialized to 1.
    /// </summary>
    /// <seealso cref="InitializerBase" />
    public class Ones : InitializerBase
    {
        /// <inheritdoc />
        internal override CNTKDictionary ToDictionary()
        {
            return CNTKLib.ConstantInitializer(1);
        }
    }
}