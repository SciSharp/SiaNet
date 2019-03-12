namespace SiaNet.Constraints
{
    using SiaNet.Engine;

    /// <summary>
    /// Base class for Constraints. Functions from the constraints module allow setting constraints (eg. non-negativity) on network parameters during optimization.
    /// The penalties are applied on a per-layer basis.The exact API will depend on the layer, but the layers Dense, Conv1D, Conv2D and Conv3D have a unified API.
    /// </summary>
    public abstract class BaseConstraint
    {
        internal IBackend K = Global.CurrentBackend;

        internal abstract Tensor Call(Tensor w);
    }
}
