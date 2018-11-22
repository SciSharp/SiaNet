using CNTK;

namespace SiaNet.Layers.Activations
{
    public class Tanh : ActivationBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Tanh(inputFunction);
        }
    }
}