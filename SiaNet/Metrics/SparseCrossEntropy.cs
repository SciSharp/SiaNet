using CNTK;

namespace SiaNet.Metrics
{
    public class SparseCrossEntropy : MetricFunction
    {
        public SparseCrossEntropy() : base(SparseCrossEntropyFunction)
        {
        }

        /// <summary>
        /// Sparses the cross entropy.
        /// </summary>
        protected static Data.Function SparseCrossEntropyFunction(Data.Variable labels, Data.Variable predictions)
        {
            return CNTKLib.CrossEntropyWithSoftmax(predictions,
                CNTKLib.Reshape(labels, new[] { labels.Shape.TotalSize }));
        }
    }
}