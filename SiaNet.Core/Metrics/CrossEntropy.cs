using CNTK;

namespace SiaNet.Metrics
{
    public class CrossEntropy : MetricFunction
    {
        public CrossEntropy() : base(CrossEntropyFunction)
        {
        }

        /// <summary>
        /// Crosses the entropy.
        /// </summary>
        protected static Data.Function CrossEntropyFunction(Data.Variable labels, Data.Variable predictions)
        {
            return CNTKLib.CrossEntropyWithSoftmax(predictions, labels);
        }
    }
}