namespace SiaNet.Constraints
{
    using SiaNet.Engine;

    /// <summary>
    /// 
    /// </summary>
    public abstract class BaseConstraint
    {
        internal IBackend K = Global.CurrentBackend;

        internal abstract Tensor Call(Tensor w);
    }
}
