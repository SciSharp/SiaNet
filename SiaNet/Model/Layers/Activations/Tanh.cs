using CNTK;

namespace SiaNet.Model.Layers.Activations
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