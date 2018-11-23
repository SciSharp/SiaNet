using CNTK;

namespace SiaNet.Layers.Activations
{
    public class ELU : ActivationBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.ELU(inputFunction);
        }
    }
}