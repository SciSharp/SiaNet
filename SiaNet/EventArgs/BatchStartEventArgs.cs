namespace SiaNet.EventArgs
{
    public class BatchStartEventArgs : System.EventArgs
    {
        public BatchStartEventArgs(uint epoch, ulong batch)
        {
            Epoch = epoch;
            Batch = batch;
        }

        public ulong Batch { get; }

        public uint Epoch { get; }
    }
}