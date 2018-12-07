using System;
using SiaNet.Backend.Interop;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed class NDListHandle : DisposableMXNetObject
    {

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="NDListHandle"/> class.
        /// </summary>
        /// <param name="handle">The handle of the MXAPINDList.</param>
        internal NDListHandle(IntPtr handle)
        {
            this.NativePtr = handle;
        }

        #endregion

        #region Methods

        #region Overrides

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            if (NativeMethods.MXNDListFree(this.NativePtr) == NativeMethods.Error)
                throw new ApplicationException($"Failed to release {nameof(NDListHandle)}");
        }

        #endregion

        #endregion

    }

}
