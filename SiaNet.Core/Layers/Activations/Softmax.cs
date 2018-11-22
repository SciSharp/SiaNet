using CNTK;

namespace SiaNet.Layers.Activations
{
    public class Softmax : ActivationBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Softmax(inputFunction);
        }
    }
}