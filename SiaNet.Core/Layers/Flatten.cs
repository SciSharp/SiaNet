using CNTK;

namespace SiaNet.Layers
{
    /// <summary>
    ///     Flattens an output
    /// </summary>
    public class Flatten : OptimizableLayerBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Reshape(inputFunction, new[] {-1});
        }
    }
}