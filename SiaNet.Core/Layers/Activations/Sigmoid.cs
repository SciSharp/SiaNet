using CNTK;

namespace SiaNet.Layers.Activations
{
    public class Sigmoid : ActivationBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Sigmoid(inputFunction);
        }
    }
}