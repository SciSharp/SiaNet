using SiaNet.Data;

namespace SiaNet.Metrics
{
    public class MeanAbsolutePercentageError : MetricFunction
    {
        public MeanAbsolutePercentageError() : base(MeanAbsolutePercentageErrorFunction)
        {
        }

        /// <summary>
        ///     Means the abs percentage error.
        /// </summary>
        protected static Function MeanAbsolutePercentageErrorFunction(Variable labels, Variable predictions)
        {
            return (Parameter) 100 *
                   ((predictions - labels).Abs() /
                    labels.Abs().Clip((Parameter) float.Epsilon, (Parameter) float.MaxValue)).ReduceMeanByAxes(-1);
        }
    }
}