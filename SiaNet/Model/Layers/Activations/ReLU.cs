using CNTK;

namespace SiaNet.Model.Layers.Activations
{
    public class ReLU : ActivationBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.ReLU(inputFunction);
        }
    }
}