using CNTK;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Global max pooling 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class GlobalMaxPool3D : LayerBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Pooling(inputFunction, PoolingType.Max,
                new[] {inputFunction.Shape[0], inputFunction.Shape[1], inputFunction.Shape[2]});
        }
    }
}