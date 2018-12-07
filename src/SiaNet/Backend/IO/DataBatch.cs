// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    /// <summary>
    /// Default object for holding a mini-batch of data and related information. This class cannot be inherited.
    /// </summary>
    public sealed class DataBatch
    {

        #region Properties

        public NDArray Data
        {
            get;
            internal set;
        }

        public int[] Index
        {
            get;
            internal set;
        }

        public NDArray Label
        {
            get;
            internal set;
        }

        public int PadNum
        {
            get;
            internal set;
        }

        #endregion

    }

}