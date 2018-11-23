using SiaNet.Data;

namespace SiaNet.Metrics
{
    public class Huber : MetricFunction
    {
        public Huber() : base(HuberFunction)
        {
        }

        /// <summary>
        ///     Huber loss
        /// </summary>
        protected static Function HuberFunction(Variable labels, Variable predictions)
        {
            return (((predictions - labels).Square() + (Constant)1).Sqrt() - (Constant)1).ReduceMeanByAxes(-1);
        }
    }
}