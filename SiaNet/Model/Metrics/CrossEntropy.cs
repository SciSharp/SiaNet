using CNTK;

namespace SiaNet.Model.Metrics
{
    public class CrossEntropy : MetricFunction
    {
        public CrossEntropy() : base(CrossEntropyFunction)
        {
        }

        /// <summary>
        /// Crosses the entropy.
        /// </summary>
        protected static Function CrossEntropyFunction(Variable labels, Variable predictions)
        {
            return CNTKLib.CrossEntropyWithSoftmax(predictions, labels);
        }
    }
}