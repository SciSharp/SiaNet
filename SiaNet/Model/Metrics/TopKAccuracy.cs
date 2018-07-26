using CNTK;

namespace SiaNet.Model.Metrics
{
    public class TopKAccuracy : MetricFunction
    {
        public TopKAccuracy(uint k = 5) : base((labels, predictions) => TopKAccuracyFunction(labels, predictions, k))
        {
        }

        /// <summary>
        ///     Tops the k accuracy.
        /// </summary>
        private static Function TopKAccuracyFunction(Variable labels, Variable predictions, uint k)
        {
            return (Constant)1f - (Function) CNTKLib.ClassificationError(predictions, labels, k);
        }
    }
}