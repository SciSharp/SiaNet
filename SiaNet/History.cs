namespace SiaNet
{
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Class to hold the neural network training information history. Useful for plotting graphs for loss and metrics. Evaluate training performance
    /// </summary>
    public class History
    {
        /// <summary>
        /// Gets or sets the train loss history.
        /// </summary>
        /// <value>
        /// The train loss.
        /// </value>
        public List<float> TrainLosses { get; set; }

        /// <summary>
        /// Gets or sets the train metric history.
        /// </summary>
        /// <value>
        /// The train metric.
        /// </value>
        public List<float> TrainMetrics { get; set; }

        /// <summary>
        /// Gets or sets the value losess.
        /// </summary>
        /// <value>
        /// The value losess.
        /// </value>
        public List<float> ValLosses { get; set; }

        /// <summary>
        /// Gets or sets the value metrics.
        /// </summary>
        /// <value>
        /// The value metrics.
        /// </value>
        public List<float> ValMetrics { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="History"/> class.
        /// </summary>
        public History()
        {
            TrainLosses = new List<float>();
            TrainMetrics = new List<float>();
            ValLosses = new List<float>();
            ValMetrics = new List<float>();
        }

        /// <summary>
        /// Adds the specified train loss.
        /// </summary>
        /// <param name="trainLoss">The train loss.</param>
        /// <param name="trainMetric">The train metric.</param>
        /// <param name="valLoss">The value loss.</param>
        /// <param name="valMetric">The value metric.</param>
        public void Add(List<float> trainLoss, List<float> trainMetric, List<float> valLoss, List<float> valMetric)
        {
            TrainLosses.Add(trainLoss.Average());
            TrainMetrics.Add(trainMetric.Average());

            if (valLoss.Count > 0)
                ValLosses.Add(valLoss.Average());

            if (valMetric.Count > 0)
                ValMetrics.Add(valMetric.Average());
        }
    }
}
