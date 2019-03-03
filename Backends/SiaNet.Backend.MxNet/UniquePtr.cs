using System;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed class UniquePtr<T> : IDisposable
    {

        #region Constructors

        public UniquePtr(T obj)
        {
            this.Ptr = obj;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets a value indicating whether this object is already disposed.
        /// </summary>
        public bool IsDisposed
        {
            get;
            private set;
        }

        private bool _Owner = true;

        public bool IsOwner => this._Owner;

        public T Ptr
        {
            get;
        }

        #endregion

        #region Methods

        public static void Move(UniquePtr<T> source, out UniquePtr<T> target)
        {
            target = new UniquePtr<T>(source.Ptr);

            source._Owner = false;
            target._Owner = true;
        }

        #endregion

        #region IDisposable Members

        /// <summary>
        /// Releases all resources used by this <see cref="DisposableMXNetObject"/>.
        /// </summary>
        public void Dispose()
        {
            GC.SuppressFinalize(this);
            this.Dispose(true);
        }

        /// <summary>
        /// Releases all resources used by this <see cref="DisposableMXNetObject"/>.
        /// </summary>
        /// <param name="disposing">Indicate value whether <see cref="IDisposable.Dispose"/> method was called.</param>
        private void Dispose(bool disposing)
        {
            if (this.IsDisposed)
            {
                return;
            }

            this.IsDisposed = true;

            if (disposing)
            {
                if (this._Owner)
                    (this.Ptr as IDisposable)?.Dispose();
            }
        }

        #endregion

    }

}
