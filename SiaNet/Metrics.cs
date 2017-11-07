namespace SiaNet
{
    using CNTK;
    using SiaNet.Common;
    using System;

    /// <summary>
    /// A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the  metrics parameter when a model is compiled. 
    /// A metric function is similar to an loss function, except that the results from evaluating a metric are not used when training the model.
    /// <see cref="OptMetrics"/>
    /// </summary>
    internal class Metrics
    {
        internal static Function Get(string metric, Variable labels, Variable predictions)
        {
            switch (metric.Trim().ToLower())
            {
                case OptMetrics.Accuracy:
                    return Accuracy(labels, predictions);
                case OptMetrics.MAE:
                    return MAE(labels, predictions);
                case OptMetrics.MAPE:
                    return MAPE(labels, predictions);
                case OptMetrics.MSE:
                    return MSE(labels, predictions);
                case OptMetrics.MSLE:
                    return MSLE(labels, predictions);
                case OptMetrics.TopKAccuracy:
                    return TopKAccuracy(labels, predictions);
                default:
                    throw new NotImplementedException(string.Format("{0} is not implemented", metric));
            }
        }

        /// <summary>
        /// Accuracies the specified labels.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function Accuracy(Variable labels, Variable predictions)
        {
            return CNTKLib.ClassificationError(predictions, labels);
        }

        /// <summary>
        /// Tops the k accuracy.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="k">The k.</param>
        /// <returns>Function.</returns>
        private static Function TopKAccuracy(Variable labels, Variable predictions, uint k=5)
        {
            return CNTKLib.ClassificationError(predictions, labels, k);
        }

        /// <summary>
        /// Mean Squared Error of the specified labels.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function MSE(Variable labels, Variable predictions)
        {
            return Losses.MeanSquaredError(labels, predictions);
        }

        /// <summary>
        /// Mean Absolute Error of the specified labels.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function MAE(Variable labels, Variable predictions)
        {
            return Losses.MeanAbsError(labels, predictions);
        }

        /// <summary>
        /// Mean Absolute Percentage Error the specified labels.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function MAPE(Variable labels, Variable predictions)
        {
            return Losses.MeanAbsPercentageError(labels, predictions);
        }

        /// <summary>
        /// Mean Squared Log Error of the specified labels.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function MSLE(Variable labels, Variable predictions)
        {
            return Losses.MeanSquaredLogError(labels, predictions);
        }

    }
}
