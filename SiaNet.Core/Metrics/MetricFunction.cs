using SiaNet.Data;
using System;

namespace SiaNet.Metrics
{
    public class MetricFunction
    {
        public delegate Function CompareFunction(Variable labels, Variable predictions);

        protected readonly CompareFunction LabelPredictionFunction;

        public MetricFunction(CompareFunction labelPredictionFunction)
        {
            LabelPredictionFunction = labelPredictionFunction;
        }

        internal CNTK.Function ToFunction(Variable labels, Variable predictions)
        {
            var function = LabelPredictionFunction?.Invoke(labels, predictions);

            if (ReferenceEquals(function, null))
            {
                throw new InvalidOperationException();
            }

            return function;
        }
    }
}