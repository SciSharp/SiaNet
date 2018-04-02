namespace SiaNet.EventArgs
{
    public class BatchEndEventArgs
    {
        public BatchEndEventArgs(
            uint epoch,
            ulong batch,
            ulong samplesSeen,
            double loss,
            double metric)
        {
            Epoch = epoch;
            Batch = batch;
            SamplesSeen = samplesSeen;
            Loss = loss;
            Metric = metric;
        }

        public ulong Batch { get; }

        public uint Epoch { get; }
        public double Loss { get; }
        public double Metric { get; }
        public ulong SamplesSeen { get; }
    }
}