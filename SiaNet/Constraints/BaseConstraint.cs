namespace SiaNet.Constraints
{
    using SiaNet.Engine;

    /// <summary>
    /// Base class for Constraints
    /// </summary>
    public abstract class BaseConstraint
    {
        internal IBackend K = Global.CurrentBackend;

        internal abstract Tensor Call(Tensor w);
    }
}
