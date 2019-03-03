using System;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    /// <summary>
    /// This is an abstract class.
    /// </summary>
    public abstract class MXNetSharedObject
    {

        #region Fields

        private int _RefCount = 1;

        #endregion

        #region Properties

        /// <summary>
        /// Native pointer of MXNet object
        /// </summary>
        public IntPtr Handle
        {
            get;
            set;
        }

        #endregion

        #region Methods

        public void AddRef()
        {
            this._RefCount++;
        }

        public void ReleaseRef()
        {
            this._RefCount--;

            if (this._RefCount == 0)
            {
                this.DisposeManaged();
                this.DisposeUnmanaged();
            }
        }

        #region Overrides

        protected virtual void DisposeManaged()
        {
            
        }

        protected virtual void DisposeUnmanaged()
        {

        }

        #endregion

        #endregion

    }

}
