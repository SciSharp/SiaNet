using SiaNet.Data;

namespace SiaNet.Metrics
{
    public class Poisson : MetricFunction
    {
        public Poisson() : base(PoissonFunction)
        {
        }

        /// <summary>
        ///     Poissons the specified labels.
        /// </summary>
        protected static Function PoissonFunction(Variable labels, Variable predictions)
        {
            return (predictions - labels * (predictions + (Parameter) float.Epsilon).Log()).ReduceMeanByAxes(-1);
        }
    }
}