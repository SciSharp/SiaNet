using CNTK;
using SiaNet.NN;

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
            return Convolution.GlobalMaxPool3D(inputFunction);
        }
    }
}