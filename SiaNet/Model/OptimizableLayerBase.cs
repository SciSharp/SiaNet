using CNTK;

namespace SiaNet.Model
{
    public abstract class OptimizableLayerBase : LayerBase
    {
        internal Function ToFunction(Shape shape)
        {
            return ToFunction(Variable.InputVariable(shape.ToNDShape(), DataType.Float));
        }
    }
}