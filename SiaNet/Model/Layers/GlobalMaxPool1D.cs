using SiaNet.NN;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Global max pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class GlobalMaxPool1D : LayerBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return Convolution.GlobalMaxPool1D(inputFunction);
        }
    }
}