using CNTK;

namespace SiaNet.Layers
{
    /// <summary>
    ///     Global average pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class GlobalAvgPool1D : LayerBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Pooling(inputFunction, PoolingType.Average, new[] {inputFunction.Shape[0]});
        }
    }
}