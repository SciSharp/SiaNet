using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    public class BatchStartEventArgs : System.EventArgs
    {
        public BatchStartEventArgs(int epoch, int batch)
        {
            Epoch = epoch;
            Batch = batch;
        }

        public int Batch { get; }

        public int Epoch { get; }
    }

    public class BatchEndEventArgs
    {
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

        public long Batch { get; }
        public int Epoch { get; }
        public float Loss { get; }
        public float Metric { get; }
    }

    public class EpochStartEventArgs : System.EventArgs
    {
        public EpochStartEventArgs(int epoch)
        {
            Epoch = epoch;
        }

        public int Epoch { get; }
    }

    public class EpochEndEventArgs : System.EventArgs
    {
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

        public int Epoch { get; }
        public float Loss { get; }
        public float Metric { get; }
        public long SamplesSeen { get; }
        public float ValidationLoss { get; }
        public float ValidationMetric { get; }
        public long Duration { get; }

    }

    public class TrainingEndEventArgs : System.EventArgs
    {
        public TrainingEndEventArgs(
            History history,
            long duration)
        {
            History = history;
            Duration = duration;
        }

        public History History { get; }

        public long Duration { get; }
    }
}
