namespace SiaNet.Model.Metrics
{
    public class MeanSquaredError : MetricFunction
    {
        public MeanSquaredError() : base(MeanSquaredErrorFunction)
        {
        }

        /// <summary>
        ///     Means the squared error.
        /// </summary>
        protected static Function MeanSquaredErrorFunction(Variable labels, Variable predictions)
        {
            return (predictions - labels).Square().ReduceMeanByAxes(-1);
        }
    }
}