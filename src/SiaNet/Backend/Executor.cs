using System;
using System.Collections.Generic;
using System.Linq;
using SiaNet.Backend.Interop;
using mx_uint = System.UInt32;
using ExecutorHandle = System.IntPtr;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed class Executor : DisposableMXNetObject
    {

        #region Fields

        private readonly Symbol _Symbol;

        #endregion

        #region Constructors

        public Executor(Symbol symbol,
                        Context context,
                        IList<NDArray> argmentArrays,
                        IList<NDArray> gradientArrays,
                        IList<OpReqType> gradReqs,
                        IList<NDArray> auxiliaryArrays)
            : this(symbol, context, argmentArrays, gradientArrays, gradReqs, auxiliaryArrays, new Dictionary<string, Context>(), null)
        {
        }

        public Executor(Symbol symbol,
                        Context context,
                        IList<NDArray> argmentArrays,
                        IList<NDArray> gradientArrays,
                        IList<OpReqType> gradReqs,
                        IList<NDArray> auxiliaryArrays,
                        IDictionary<string, Context> groupToCtx)
            : this(symbol, context, argmentArrays, gradientArrays, gradReqs, auxiliaryArrays, groupToCtx, null)
        {
        }

        public Executor(Symbol symbol,
                        Context context,
                        IList<NDArray> argmentArrays,
                        IList<NDArray> gradientArrays,
                        IList<OpReqType> gradReqs,
                        IList<NDArray> auxiliaryArrays,
                        IDictionary<string, Context> groupToCtx,
                        Executor sharedExec)
        {
            if (symbol == null)
                throw new ArgumentNullException(nameof(symbol));
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (argmentArrays == null)
                throw new ArgumentNullException(nameof(argmentArrays));
            if (gradientArrays == null)
                throw new ArgumentNullException(nameof(gradientArrays));
            if (gradReqs == null)
                throw new ArgumentNullException(nameof(gradReqs));
            if (auxiliaryArrays == null)
                throw new ArgumentNullException(nameof(auxiliaryArrays));
            if (groupToCtx == null)
                throw new ArgumentNullException(nameof(groupToCtx));

            this.ArgmentArrays = argmentArrays;
            this.GradientArrays = gradientArrays;
            this.AuxiliaryArrays = auxiliaryArrays;
            this._Symbol = symbol;

            var argHandles = argmentArrays.Select(array => array.GetHandle()).ToArray();
            var gradHandles = gradientArrays.Select(array => array.GetHandle()).ToArray();
            var auxHandles = auxiliaryArrays.Select(array => array.GetHandle()).ToArray();
            var gradReqsUint = gradReqs.Select(s => (mx_uint)s).ToArray();

            var mapKeys = new string[groupToCtx.Count];
            var devTypes = new int[groupToCtx.Count];
            var devIds = new int[groupToCtx.Count];
            var keys = groupToCtx.Keys.ToArray();
            for (var index = 0; index < keys.Length; index++)
            {
                var key = keys[index];
                mapKeys[index] = key;
                var value = groupToCtx[key];
                devTypes[index] = (int)value.GetDeviceType();
                devIds[index] = value.GetDeviceId();
            }

            var sharedExecHandle = sharedExec?.Handle ?? IntPtr.Zero;

            Logging.CHECK_EQ(NativeMethods.MXExecutorBindEX(symbol.GetHandle(),
                                                           (int)context.GetDeviceType(),
                                                           context.GetDeviceId(),
                                                           (uint)groupToCtx.Count,
                                                           mapKeys,
                                                           devTypes,
                                                           devIds,
                                                           (uint)argHandles.Length,
                                                           argHandles,
                                                           gradHandles,
                                                           gradReqsUint,
                                                           (uint)auxHandles.Length,
                                                           auxHandles,
                                                           sharedExecHandle,
                                                           out var handle), NativeMethods.OK);
            this.Handle = handle;

            this.Outputs = new List<NDArray>();
            Logging.CHECK_EQ(NativeMethods.MXExecutorOutputs(this.Handle, out var outSize, out var outArray), 0);
            var outArrayArray = InteropHelper.ToPointerArray(outArray, outSize);
            for (mx_uint i = 0; i < outSize; ++i)
                this.Outputs.Add(new NDArray(outArrayArray[i]));
        }

        public Executor(ExecutorHandle h)
        {
            if (h == IntPtr.Zero)
                throw new ArgumentException("Can not pass IntPtr.Zero", nameof(h));

            this.Handle = h;
        }

        #endregion

        #region Properties

        internal ExecutorHandle Handle
        {
            get;
        }

        public IList<NDArray> Outputs
        {
            get;
        }

        public IList<NDArray> ArgmentArrays
        {
            get;
        }

        public IList<NDArray> GradientArrays
        {
            get;
        }

        public IList<NDArray> AuxiliaryArrays
        {
            get;
        }

        #endregion

        #region Methods

        public IDictionary<string, NDArray> ArgmentDictionary()
        {
            return GetDictionary(this._Symbol.ListArguments(), this.ArgmentArrays);
        }

        public IDictionary<string, NDArray> GradientDictionary()
        {
            return GetDictionary(this._Symbol.ListArguments(), this.GradientArrays);
        }

        public IDictionary<string, NDArray> AuxiliaryDictionary()
        {
            return GetDictionary(this._Symbol.ListAuxiliaryStates(), this.AuxiliaryArrays);
        }

        public void Backward()
        {
            this.Backward(new List<NDArray>());
        }

        public void Backward(IList<NDArray> headGrads)
        {
            if (headGrads == null)
                throw new ArgumentNullException(nameof(headGrads));

            var tmp = headGrads.Select(d => d.GetHandle()).ToArray();
            if (tmp.Length > 0)
                NativeMethods.MXExecutorBackward(this.Handle, (uint) tmp.Length, tmp);
            else
                NativeMethods.MXExecutorBackward(this.Handle, 0, null);
        }

        public void Forward(bool isTrain)
        {
            NativeMethods.MXExecutorForward(this.Handle, isTrain ? 1 : 0);
            Logging.CHECK_EQ(NativeMethods.MXExecutorOutputs(this.Handle, out var outSize, out var outArray), 0);
            var outArrayArray = InteropHelper.ToPointerArray(outArray, outSize);
            for (var i = 0; i < outSize; ++i)
            {
                this.Outputs[i]?.Dispose();
                this.Outputs[i] = new NDArray(outArrayArray[i]);
            }
        }

        #region Overrids

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            NativeMethods.MXExecutorFree(this.Handle);
        }

        #endregion

        #region Helpers

        private static IDictionary<string, NDArray> GetDictionary(IList<string> names, IList<NDArray> arrays)
        {
            var ret = new Dictionary<string, NDArray>();

            var set = new HashSet<string>();
            foreach (var s in names)
            {
                Logging.CHECK(set.Contains(s), $"Duplicate names detected, {s}");
                set.Add(s);
            }

            Logging.CHECK_EQ(set.Count, arrays.Count, "names size not equal to arrays size");
            for (var i = 0; i < names.Count; ++i)
                ret[names[i]] = arrays[i];

            return ret;
        }

        #endregion

        #endregion
        
    }

}
