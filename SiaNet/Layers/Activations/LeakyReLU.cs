using CNTK;

namespace SiaNet.Layers.Activations
{
    public class LeakyReLU : ActivationBase
    {
        public LeakyReLU(double alpha = 0.1)
        {
            Alpha = alpha;
        }

        public double Alpha { get; protected set; }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.LeakyReLU(inputFunction, Alpha);
        }
    }
}