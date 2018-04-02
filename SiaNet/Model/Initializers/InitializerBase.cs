using CNTK;

namespace SiaNet.Model.Initializers
{
    /// <summary>
    ///     Base for the initializer
    /// </summary>
    public abstract class InitializerBase
    {
        internal abstract CNTKDictionary ToDictionary();
    }
}