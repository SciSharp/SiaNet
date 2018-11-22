namespace SiaNet.EventArgs
{
    public class BatchStartEventArgs : System.EventArgs
    {
        public BatchStartEventArgs(uint epoch, long batch)
        {
            Epoch = epoch;
            Batch = batch;
        }

        public long Batch { get; }

        public uint Epoch { get; }
    }
}