using CNTK;

namespace SiaNet.Layers
{
    /// <summary>
    ///     Global max pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class GlobalMaxPool2D : LayerBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Pooling(inputFunction, PoolingType.Max,
                new[] {inputFunction.Shape[0], inputFunction.Shape[1]});
        }
    }
}