using CNTK;

namespace SiaNet.Model.Metrics
{
    public class SparseCrossEntropy : MetricFunction
    {
        public SparseCrossEntropy() : base(SparseCrossEntropyFunction)
        {
        }

        /// <summary>
        /// Sparses the cross entropy.
        /// </summary>
        protected static Function SparseCrossEntropyFunction(Variable labels, Variable predictions)
        {
            return CNTKLib.CrossEntropyWithSoftmax(predictions,
                CNTKLib.Reshape(labels, new[] { labels.Shape.TotalSize }));
        }
    }
}