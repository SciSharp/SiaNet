using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using SiaNet.Backend.MxNetLib.Extensions;
using SiaNet.Backend.MxNetLib.Interop;
using SymbolHandle = System.IntPtr;
using NDArrayHandle = System.IntPtr;
using mx_uint = System.UInt32;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed class Operator : DisposableMXNetObject
    {

        #region Fields

        private static readonly OpMap OpMap;

        private readonly Dictionary<string, string> _Params = new Dictionary<string, string>();

        private readonly List<SymbolHandle> _InputSymbols = new List<SymbolHandle>();

        private readonly List<NDArrayHandle> _InputNdarrays = new List<NDArrayHandle>();

        private readonly List<string> _InputKeys = new List<string>();

        private readonly List<string> _ArgNames = new List<string>();

        private readonly SymbolHandle _Handle;

        #endregion

        #region Constructors

        static Operator()
        {
            OpMap = new OpMap();
        }

        public Operator(SymbolHandle handle)
        {
            this.NativePtr = handle;
        }

        public Operator(string operatorName)
        {
            this._OpName = operatorName;
            this._Handle = OpMap.GetSymbolCreator(operatorName);
            
            var return_type = System.IntPtr.Zero;
            Logging.CHECK_EQ(NativeMethods.MXSymbolGetAtomicSymbolInfo(this._Handle,
                                                                       out var name,
                                                                       out var description,
                                                                       out var numArgs,
                                                                       out var argNames,
                                                                       out var argTypeInfos,
                                                                       out var argDescriptions,
                                                                       out var keyVarNumArgs,
                                                                       ref return_type), NativeMethods.OK);

            var argNamesArray = InteropHelper.ToPointerArray(argNames, numArgs);
            for (var i = 0; i < numArgs; ++i)
            {
                var pArgName = argNamesArray[i];
                this._ArgNames.Add(Marshal.PtrToStringAnsi(pArgName));
            }
        }

        #endregion

        #region Properties

        public SymbolHandle Handle => this.NativePtr;

        private readonly string _OpName;

        public string Name
        {
            get
            {
                this.ThrowIfDisposed();
                return this._OpName;
            }
        }

        #endregion

        #region Methods

        public Symbol CreateSymbol(string name = "")
        {
            if (this._InputKeys.Count > 0)
                Logging.CHECK_EQ(this._InputKeys.Count, this._InputSymbols.Count);

            var pname = name == "" ? null : name;

            var keys = this._Params.Keys.ToArray();
            var paramKeys = new string[keys.Length];
            var paramValues = new string[keys.Length];
            for (var index = 0; index < keys.Length; index++)
            {
                var key = keys[index];
                paramKeys[index] = key;
                paramValues[index] = this._Params[key];
            }

            var inputKeys = this._InputKeys.Count != 0 ? this._InputKeys.ToArray() : null;

            Logging.CHECK_EQ(NativeMethods.MXSymbolCreateAtomicSymbol(this._Handle,
                                                                      (uint)paramKeys.Length,
                                                                      paramKeys,
                                                                      paramValues,
                                                                      out var symbolHandle), NativeMethods.OK);

            Logging.CHECK_EQ(NativeMethods.MXSymbolCompose(symbolHandle,
                                                           pname,
                                                           (uint)this._InputSymbols.Count,
                                                           inputKeys,
                                                           this._InputSymbols.ToArray()), NativeMethods.OK);

            return new Symbol(symbolHandle);
        }

        public List<NDArray> Invoke()
        {
            var outputs = new List<NDArray>();
            this.Invoke(outputs);
            return outputs;
        }

        public void Invoke(NDArray output)
        {
            if (output == null)
                throw new ArgumentNullException(nameof(output));

            var outputs = new List<NDArray>(new[] { output });
            this.Invoke(outputs);
        }

        public void Invoke(List<NDArray> outputs)
        {
            if (outputs == null)
                throw new ArgumentNullException(nameof(outputs));

            if (this._InputKeys.Count > 0)
                Logging.CHECK_EQ(this._InputKeys.Count, this._InputSymbols.Count);

            var keys = this._Params.Keys.ToArray();
            var paramKeys = new string[keys.Length];
            var paramValues = new string[keys.Length];
            for (var index = 0; index < keys.Length; index++)
            {
                var key = keys[index];
                paramKeys[index] = key;
                paramValues[index] = this._Params[key];
            }

            var num_inputs = this._InputNdarrays.Count;
            var num_outputs = outputs.Count;

            var output_handles = outputs.Select(array => array.NativePtr).ToArray();
            NDArrayHandle[] outputsReceiver = null;
            if (num_outputs > 0)
                outputsReceiver = output_handles;

            Logging.CHECK_EQ(NativeMethods.MXImperativeInvoke(this._Handle,
                                                              num_inputs,
                                                              this._InputNdarrays.ToArray(),
                                                              ref num_outputs,
                                                              ref outputsReceiver,
                                                              paramKeys.Length,
                                                              paramKeys,
                                                              paramValues), NativeMethods.OK);

            if (outputs.Count > 0)
                return;

            outputs.AddRange(outputsReceiver.Select(ptr => new NDArray(ptr)));
        }

        public void PushInput(Symbol symbol)
        {
            if (symbol == null)
                throw new ArgumentNullException(nameof(symbol));

            this._InputSymbols.Add(symbol.GetHandle());
        }

        public Operator PushInput(NDArray ndarray)
        {
            if (ndarray == null)
                throw new ArgumentNullException(nameof(ndarray));

            this._InputNdarrays.Add(ndarray.GetHandle());
            return this;
        }

        public Operator Set(params Object[] args)
        {
            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                if (arg is Symbol)
                    SetParam(i, (Symbol)arg);
                else if (arg is NDArray)
                    SetParam(i, (NDArray)arg);
                else if (arg is IEnumerable<Symbol>)
                    SetParam(i, (IEnumerable<Symbol>)arg);
                else
                    SetParam(i, arg);
            }
            return this;
        }

        public Operator SetInput(string name, Symbol symbol)
        {
            this._InputKeys.Add(name);
            this._InputSymbols.Add(symbol.GetHandle());
            return this;
        }

        public Operator SetInput(string name, NDArray ndarray)
        {
            this._InputKeys.Add(name);
            this._InputNdarrays.Add(ndarray.NativePtr);
            return this;
        }

        public Operator SetParam(string key, object value)
        {
            this._Params[key] = value.ToValueString();
            return this;
        }

        public Operator SetParam(int pos, NDArray val)
        {
            this._InputNdarrays.Add(val.NativePtr);
            return this;
        }

        public Operator SetParam(int pos, Symbol val)
        {
            this._InputSymbols.Add(val.GetHandle());
            return this;
        }

        public Operator SetParam(int pos, IEnumerable<Symbol> val)
        {
            this._InputSymbols.AddRange(val.Select(s => s.GetHandle()));
            return this;
        }

        public Operator SetParam(int pos, object val)
        {
            this._Params[this._ArgNames[pos]] = val.ToString();
            return this;
        }

        #region Overrides

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            NativeMethods.MXSymbolFree(this.NativePtr);
        }

        #endregion

        #endregion

    }

}
