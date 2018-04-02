using SiaNet.NN;

namespace SiaNet.Model.Layers
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
            return Convolution.GlobalMaxPool2D(inputFunction);
        }
    }
}