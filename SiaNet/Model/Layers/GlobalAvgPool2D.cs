using SiaNet.NN;

namespace SiaNet.Model.Layers
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
            return Convolution.GlobalAvgPool2D(inputFunction);
        }
    }
}