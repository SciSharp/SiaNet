using CNTK;

namespace SiaNet.Layers
{
    /// <summary>
    ///     Global average pooling operation for 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class GlobalAvgPool3D : LayerBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Pooling(inputFunction, PoolingType.Average,
                new[] {inputFunction.Shape[0], inputFunction.Shape[1], inputFunction.Shape[2]});
        }
    }
}