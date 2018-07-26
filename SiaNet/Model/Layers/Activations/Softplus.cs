using CNTK;

namespace SiaNet.Model.Layers.Activations
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