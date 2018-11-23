using CNTK;

namespace SiaNet.Metrics
{
    public class TopKAccuracy : MetricFunction
    {
        public TopKAccuracy(uint k = 5) : base((labels, predictions) => TopKAccuracyFunction(labels, predictions, k))
        {
        }

        /// <summary>
        ///     Tops the k accuracy.
        /// </summary>
        private static Function TopKAccuracyFunction(Data.Variable labels, Data.Variable predictions, uint k)
        {
            return (Data.Constant)1f - (Data.Function) CNTKLib.ClassificationError(predictions, labels, k);
        }
    }
}