using CNTK;
using SiaNet.NN;

namespace SiaNet.Model.Layers
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
            return Convolution.GlobalAvgPool3D(inputFunction);
        }
    }
}