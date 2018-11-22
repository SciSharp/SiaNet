namespace SiaNet.EventArgs
{
    public class EpochStartEventArgs : System.EventArgs
    {
        public EpochStartEventArgs(uint epoch)
        {
            Epoch = epoch;
        }

        public uint Epoch { get; }
    }
}