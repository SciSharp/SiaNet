namespace SiaNet
{
    /// <summary>
    /// Event triggered when batch training starts
    /// </summary>
    /// <seealso cref="System.EventArgs" />
    public class BatchStartEventArgs : System.EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BatchStartEventArgs"/> class.
        /// </summary>
        /// <param name="epoch">The current epoch number.</param>
        /// <param name="batch">The current batch number.</param>
        public BatchStartEventArgs(int epoch, int batch)
        {
            Epoch = epoch;
            Batch = batch;
        }

        /// <summary>
        /// Gets the current batch number.
        /// </summary>
        /// <value>
        /// The batch.
        /// </value>
        public int Batch { get; }

        /// <summary>
        /// Gets the current epoch number.
        /// </summary>
        /// <value>
        /// The epoch.
        /// </value>
        public int Epoch { get; }
    }

    /// <summary>
    /// Event triggered during batch training ends
    /// </summary>
    public class BatchEndEventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BatchEndEventArgs"/> class.
        /// </summary>
        /// <param name="epoch">The current epoch number.</param>
        /// <param name="batch">The current batch number.</param>
        /// <param name="loss">The loss value for the batch.</param>
        /// <param name="metric">The metric value for the batch.</param>
        public BatchEndEventArgs(
            int epoch,
            long batch,
            float loss,
            float metric)
        {
            Epoch = epoch;
            Batch = batch;
            Loss = loss;
            Metric = metric;
        }

        /// <summary>
        /// Gets the current batch number.
        /// </summary>
        /// <value>
        /// The batch.
        /// </value>
        public long Batch { get; }

        /// <summary>
        /// Gets the current epoch number.
        /// </summary>
        /// <value>
        /// The epoch.
        /// </value>
        public int Epoch { get; }

        /// <summary>
        /// Gets the loss value for this batch.
        /// </summary>
        /// <value>
        /// The loss.
        /// </value>
        public float Loss { get; }

        /// <summary>
        /// Gets the metric value for this batch.
        /// </summary>
        /// <value>
        /// The metric.
        /// </value>
        public float Metric { get; }
    }

    /// <summary>
    /// Event triggered when the training epoch starts
    /// </summary>
    /// <seealso cref="System.EventArgs" />
    public class EpochStartEventArgs : System.EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EpochStartEventArgs"/> class.
        /// </summary>
        /// <param name="epoch">The current epoch number.</param>
        public EpochStartEventArgs(int epoch)
        {
            Epoch = epoch;
        }

        /// <summary>
        /// Gets the current epoch.
        /// </summary>
        /// <value>
        /// The epoch.
        /// </value>
        public int Epoch { get; }
    }

    /// <summary>
    /// Event triggered when the training epoch ends
    /// </summary>
    /// <seealso cref="System.EventArgs" />
    public class EpochEndEventArgs : System.EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EpochEndEventArgs"/> class.
        /// </summary>
        /// <param name="epoch">The current epoch number.</param>
        /// <param name="samplesSeen">The number of samples seen during this training for the current epoch.</param>
        /// <param name="loss">The loss value for the current epoch.</param>
        /// <param name="validationLoss">The validation loss for the epoch.</param>
        /// <param name="metric">The metric value for the epoch.</param>
        /// <param name="validationMetric">The validation metric value for the epoch.</param>
        /// <param name="duration">The time took to complete the epoch. In milliseconds</param>
        public EpochEndEventArgs(
            int epoch,
            long samplesSeen,
            float loss,
            float validationLoss,
            float metric,
            float validationMetric,
            long duration)
        {
            Epoch = epoch;
            SamplesSeen = samplesSeen;
            Loss = loss;
            ValidationLoss = validationLoss;
            Metric = metric;
            ValidationMetric = validationMetric;
            Duration = duration;
        }

        /// <summary>
        /// Gets the current epoch.
        /// </summary>
        /// <value>
        /// The epoch.
        /// </value>
        public int Epoch { get; }

        /// <summary>
        /// Gets the loss value for the current epoch.
        /// </summary>
        /// <value>
        /// The loss.
        /// </value>
        public float Loss { get; }

        /// <summary>
        /// Gets the metric value for the current epoch.
        /// </summary>
        /// <value>
        /// The metric.
        /// </value>
        public float Metric { get; }

        /// <summary>
        /// Gets the nymber of samples seen during epoch training.
        /// </summary>
        /// <value>
        /// The samples seen.
        /// </value>
        public long SamplesSeen { get; }

        /// <summary>
        /// Gets the validation loss value for the current epoch.
        /// </summary>
        /// <value>
        /// The validation loss.
        /// </value>
        public float ValidationLoss { get; }

        /// <summary>
        /// Gets the validation metric value for the current epoch.
        /// </summary>
        /// <value>
        /// The validation metric.
        /// </value>
        public float ValidationMetric { get; }

        /// <summary>
        /// Gets the time taken to complete the epoch. In milliseconds.
        /// </summary>
        /// <value>
        /// The duration.
        /// </value>
        public long Duration { get; }

    }

    /// <summary>
    /// Event triggered when the completed model training ends
    /// </summary>
    /// <seealso cref="System.EventArgs" />
    public class TrainingEndEventArgs : System.EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TrainingEndEventArgs"/> class.
        /// </summary>
        /// <param name="history">The training history with all epoch losses and metrics for both train and validation dataset.</param>
        /// <param name="duration">The total time taken (in millisecond) to finish the model training.</param>
        public TrainingEndEventArgs(
            History history,
            long duration)
        {
            History = history;
            Duration = duration;
        }

        /// <summary>
        /// Gets the training history with all epoch losses and metrics for both train and validation dataset.
        /// </summary>
        /// <value>
        /// The history.
        /// </value>
        public History History { get; }

        /// <summary>
        /// Gets the total time taken (in millisecond) to finish the model training.
        /// </summary>
        /// <value>
        /// The duration.
        /// </value>
        public long Duration { get; }
    }
}
