namespace SiaNet.Metrics
{
    using SiaNet.Engine;

    /// <summary>
    /// A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the metrics parameter when a model is compiled.
    /// </summary>
    public abstract class BaseMetric
    {
        internal IBackend K = Global.CurrentBackend;

        /// <summary>
        /// Gets or sets the name of the metrics.
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseMetric"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        public BaseMetric(string name)
        {
            Name = name;
        }

        /// <summary>
        /// Calculates the metric with predicted and true values.
        /// </summary>
        /// <param name="preds">The predicted value.</param>
        /// <param name="labels">The true value.</param>
        /// <returns></returns>
        public abstract Tensor Calc(Tensor preds, Tensor labels);

        /// <summary>
        /// Gets the specified metric type.
        /// </summary>
        /// <param name="metricType">Type of the metric.</param>
        /// <returns></returns>
        internal static BaseMetric Get(MetricType metricType)
        {
            BaseMetric metric = null;
            switch (metricType)
            {
                case MetricType.Accuracy:
                    metric = new Accuracy();
                    break;
                case MetricType.BinaryAccurary:
                    metric = new BinaryAccuracy();
                    break;
                case MetricType.MSE:
                    metric = new MSE();
                    break;
                case MetricType.MAE:
                    metric = new MAE();
                    break;
                case MetricType.MAPE:
                    metric = new MAPE();
                    break;
                case MetricType.MSLE:
                    metric = new MSLE();
                    break;
                default:
                    break;
            }

            return metric;
        }
    }
}
