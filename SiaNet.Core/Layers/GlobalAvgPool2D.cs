using CNTK;

namespace SiaNet.Layers
{
    /// <summary>
    ///     Global average pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class GlobalAvgPool2D : LayerBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Pooling(inputFunction, PoolingType.Average,
                new[] {inputFunction.Shape[0], inputFunction.Shape[1]});
        }
    }
}