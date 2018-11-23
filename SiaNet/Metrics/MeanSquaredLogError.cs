using SiaNet.Data;

namespace SiaNet.Metrics
{
    public class MeanSquaredLogError : MetricFunction
    {
        public MeanSquaredLogError() : base(MeanSquaredLogErrorFunction)
        {
        }

        /// <summary>
        ///     Means the squared log error.
        /// </summary>
        protected static Function MeanSquaredLogErrorFunction(Variable labels, Variable predictions)
        {
            var predictionLog = predictions.Clip((Parameter) float.Epsilon, (Parameter) float.MaxValue).Log();
            var labelLog = labels.Clip((Parameter) float.Epsilon, (Parameter) float.MaxValue).Log();

            return (predictionLog - labelLog).Square().ReduceMeanByAxes(-1);
        }
    }
}