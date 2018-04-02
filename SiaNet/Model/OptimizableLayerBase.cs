using CNTK;

namespace SiaNet.Model
{
    public abstract class OptimizableLayerBase : LayerBase
    {
        internal CNTK.Function ToFunction(Shape shape)
        {
            return ToFunction(CNTK.Variable.InputVariable(shape, DataType.Float));
        }
    }
}