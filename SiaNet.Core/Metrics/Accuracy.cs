using CNTK;

namespace SiaNet.Metrics
{
    public class Accuracy : MetricFunction
    {
        /// <inheritdoc />
        public Accuracy() : base(AccuracyFunction)
        {
        }

        /// <summary>
        ///     Accuracies the specified labels.
        /// </summary>
        private static Data.Function AccuracyFunction(Data.Variable labels, Data.Variable predictions)
        {
            return (Data.Constant)1f - (Data.Function)CNTKLib.ClassificationError(predictions, labels);
        }
    }
}