using CNTK;

namespace SiaNet.Initializers
{
    /// <summary>
    ///     Base for the initializer
    /// </summary>
    public abstract class InitializerBase
    {
        internal abstract CNTKDictionary ToDictionary();
    }
}