using System;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    /// <summary>
    /// A class which has a pointer of Caffe object. This is an abstract class.
    /// </summary>
    public abstract class MXNetObject
    {

        #region Properties

        /// <summary>
        /// Native pointer of MXNet object
        /// </summary>
        public IntPtr NativePtr
        {
            get;
            protected set;
        }

        #endregion

    }

}
