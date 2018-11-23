using CNTK;

namespace SiaNet.Metrics
{
    public class BinaryCrossEntropy : MetricFunction
    {
        public BinaryCrossEntropy() : base(BinaryCrossEntropyFunction)
        {
        }

        /// <summary>
        /// Binaries the cross entropy.
        /// </summary>
        protected static Data.Function BinaryCrossEntropyFunction(Data.Variable labels, Data.Variable predictions)
        {
            return CNTKLib.BinaryCrossEntropy(predictions, labels);
        }
    }
}