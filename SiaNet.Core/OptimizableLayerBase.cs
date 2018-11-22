using CNTK;
using SiaNet.Data;

namespace SiaNet
{
    public abstract class OptimizableLayerBase : LayerBase
    {
        internal CNTK.Function ToFunction(Shape shape)
        {
            return ToFunction(CNTK.Variable.InputVariable(shape, DataType.Float));
        }
    }
}