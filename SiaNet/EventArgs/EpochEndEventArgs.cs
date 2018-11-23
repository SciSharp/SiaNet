namespace SiaNet.EventArgs
{
    public class EpochEndEventArgs : System.EventArgs
    {
        public EpochEndEventArgs(
            uint epoch,
            ulong samplesSeen,
            double loss,
            double validationLoss,
            double metric,
            double validationMetric)
        {
            Epoch = epoch;
            SamplesSeen = samplesSeen;
            Loss = loss;
            ValidationLoss = validationLoss;
            Metric = metric;
            ValidationMetric = validationMetric;
        }

        public uint Epoch { get; }
        public double Loss { get; }
        public double Metric { get; }
        public ulong SamplesSeen { get; }
        public double ValidationLoss { get; }
        public double ValidationMetric { get; }
    }
}