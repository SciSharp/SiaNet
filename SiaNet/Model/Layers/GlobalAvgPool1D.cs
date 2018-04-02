using SiaNet.NN;

namespace SiaNet.Model.Layers
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
            return Convolution.GlobalAvgPool1D(inputFunction);
        }
    }
}