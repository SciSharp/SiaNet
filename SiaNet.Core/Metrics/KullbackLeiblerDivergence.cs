using SiaNet.Data;

namespace SiaNet.Metrics
{
    public class KullbackLeiblerDivergence : MetricFunction
    {
        public KullbackLeiblerDivergence() : base(KullbackLeiblerDivergenceFunction)
        {
        }

        /// <summary>
        ///     Kullbacks the leibler divergence.
        /// </summary>
        protected static Function KullbackLeiblerDivergenceFunction(Variable labels, Variable predictions)
        {
            var clippedLabels = labels.Clip((Parameter) float.Epsilon, (Parameter) 1);
            var clippedPredictions = predictions.Clip((Parameter) float.Epsilon, (Parameter) 1);

            return (clippedLabels * (clippedLabels / clippedPredictions).Log()).ReduceSumByAxes(-1);
        }
    }
}