using CNTK;

namespace SiaNet.Layers.Activations
{
    public class SELU : ActivationBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.SELU(inputFunction);
        }
    }
}