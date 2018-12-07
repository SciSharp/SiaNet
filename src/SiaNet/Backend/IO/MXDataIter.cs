using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using SiaNet.Backend.Extensions;
using SiaNet.Backend.Interop;
using mx_float = System.Single;
using DataIterCreator = System.IntPtr;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed class MXDataIter : DataIter
    {

        #region Fields

        private static readonly MXDataIterMap DataiterMap = new MXDataIterMap();

        private readonly DataIterCreator _Creator;

        private readonly Dictionary<string, string> _Params = new Dictionary<string, string>();

        private MXDataIterBlob _BlobPtr;

        #endregion

        #region Constructors

        public MXDataIter(string mxdataiterType)
        {
            this._Creator = DataiterMap.GetMXDataIterCreator(mxdataiterType);
            this._BlobPtr = new MXDataIterBlob();
        }

        public MXDataIter(MXDataIter other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));

            this._Creator = other._Creator;
            this._Params = new Dictionary<string, string>(other._Params);

            other._BlobPtr.AddRef();
            this._BlobPtr = other._BlobPtr;
        }

        #endregion

        #region Methods

        public MXDataIter CreateDataIter()
        {
            var keys = this._Params.Keys.ToArray();
            var paramKeys = new string[keys.Length];
            var paramValues = new string[keys.Length];
            for (var i = 0; i < keys.Length; i++)
            {
                var key = keys[i];
                paramKeys[i] = key;
                paramValues[i] = this._Params[key];
            }

            NativeMethods.MXDataIterCreateIter(this._Creator,
                                               (uint)paramKeys.Length,
                                               paramKeys,
                                               paramValues,
                                               out var @out);

            if (@out == IntPtr.Zero)
            {
                var error = NativeMethods.MXGetLastError();
                var innerMessage = Marshal.PtrToStringAnsi(error);
                throw new MXNetException($"Failed to create DataIter: {innerMessage}");
            }

            this._BlobPtr.Handle = @out;
            return this;
        }

        public MXDataIter SetParam(string name, object value)
        {
            this._Params[name] = value.ToValueString();
            return this;
        }

        public override void SetBatch(uint batchSize)
        {
            SetParam("batch_size", batchSize);
            base.SetBatch(batchSize);
        }

        #region Overrides

        protected override void DisposeManaged()
        {
            base.DisposeManaged();

            this._BlobPtr?.ReleaseRef();
            this._BlobPtr = null;
        }

        #endregion

        #endregion

        #region DataIter Members

        public override void BeforeFirst()
        {
            var r = NativeMethods.MXDataIterBeforeFirst(this._BlobPtr.Handle);
            Logging.CHECK_EQ(r, 0);
        }

        public override NDArray GetData()
        {
            var r = NativeMethods.MXDataIterGetData(this._BlobPtr.Handle, out var handle);
            Logging.CHECK_EQ(r, 0);
            return new NDArray(handle);
        }

        public override int[] GetIndex()
        {
            var r = NativeMethods.MXDataIterGetIndex(this._BlobPtr.Handle, out var outIndex, out var outSize);
            Logging.CHECK_EQ(r, 0);

            var outIndexArray = InteropHelper.ToUInt64Array(outIndex, (uint)outSize);
            var ret = new int[outSize];
            for (var i = 0ul; i < outSize; ++i)
                ret[i] = (int)outIndexArray[i];

            return ret;
        }

        public override NDArray GetLabel()
        {
            var r = NativeMethods.MXDataIterGetLabel(this._BlobPtr.Handle, out var handle);
            Logging.CHECK_EQ(r, 0);
            return new NDArray(handle);
        }

        public override int GetPadNum()
        {
            var r = NativeMethods.MXDataIterGetPadNum(this._BlobPtr.Handle, out var @out);
            Logging.CHECK_EQ(r, 0);
            return @out;
        }

        public override bool Next()
        {
            var r = NativeMethods.MXDataIterNext(this._BlobPtr.Handle, out var @out);
            Logging.CHECK_EQ(r, 0);
            return @out > 0;
        }

        #endregion

    }

}
