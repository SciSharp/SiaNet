using CNTK;

namespace SiaNet.Model.Metrics
{
    public class BinaryCrossEntropy : MetricFunction
    {
        public BinaryCrossEntropy() : base(BinaryCrossEntropyFunction)
        {
        }

        /// <summary>
        /// Binaries the cross entropy.
        /// </summary>
        protected static Function BinaryCrossEntropyFunction(Variable labels, Variable predictions)
        {
            return CNTKLib.BinaryCrossEntropy(predictions, labels);
        }
    }
}