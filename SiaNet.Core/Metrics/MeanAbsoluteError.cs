using SiaNet.Data;

namespace SiaNet.Metrics
{
    public class MeanAbsoluteError : MetricFunction
    {
        public MeanAbsoluteError() : base(MeanAbsoluteErrorFunction)
        {
        }

        /// <summary>
        ///     Means the abs error.
        /// </summary>
        protected static Function MeanAbsoluteErrorFunction(Variable labels, Variable predictions)
        {
            return (predictions - labels).Abs().ReduceMeanByAxes(-1);
        }
    }
}