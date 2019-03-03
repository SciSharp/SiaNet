// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public abstract class DataIter : DisposableMXNetObject
    {

        #region Methods
        public uint BatchSize { get; set; }

        public uint LabelCount
        {
            get
            {
                return GetLabel().GetShape()[0];
            }
        }

        public abstract void BeforeFirst();

        public abstract NDArray GetData();

        public DataBatch GetDataBatch()
        {
            return new DataBatch
            {
                Data = this.GetData(),
                Label = this.GetLabel(),
                PadNum = this.GetPadNum(),
                Index = this.GetIndex()
            };
        }

        public abstract int[] GetIndex();

        public abstract NDArray GetLabel();

        public abstract int GetPadNum();

        public abstract bool Next();

        public void Reset()
        {
            this.BeforeFirst();
        }

        public virtual void SetBatch(uint batchSize)
        {
            
        }

        #endregion

    }

}