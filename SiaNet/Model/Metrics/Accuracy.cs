using CNTK;

namespace SiaNet.Model.Metrics
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
        private static Function AccuracyFunction(Variable labels, Variable predictions)
        {
            return (Constant)1f - (Function)CNTKLib.ClassificationError(predictions, labels);
        }
    }
}