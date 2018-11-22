using CNTK;

namespace SiaNet.Layers.Activations
{
    public class Softplus : ActivationBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Softplus(inputFunction);
        }
    }
}