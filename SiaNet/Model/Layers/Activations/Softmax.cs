using CNTK;

namespace SiaNet.Model.Layers.Activations
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