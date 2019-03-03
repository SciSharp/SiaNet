using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using SiaNet.Backend.MxNetLib.Interop;
using mx_uint = System.UInt32;
using SymbolHandle = System.IntPtr;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public class Symbol : DisposableMXNetObject
    {

        #region Fields

        private static readonly OpMap OpMap;

        #endregion

        #region Constructors

        static Symbol()
        {
            OpMap = new OpMap();
        }

        public Symbol()
            : this(IntPtr.Zero)
        {
        }

        public Symbol(IntPtr handle)
        {
            this.NativePtr = handle;
        }

        public Symbol(string name)
        {
            if (NativeMethods.MXSymbolCreateVariable(name, out var @out) != NativeMethods.OK)
                throw new MXNetException($"Failed to create {nameof(Symbol)}");

            this.NativePtr = @out;
        }

        //public Symbol(string operatorName, 
        //              string name,
        //              IList<string> inputKeys,
        //              IList<SymbolHandle> inputValues,
        //              IList<string> configKeys,
        //              IList<string> configValues)
        //{
        //    if (inputKeys == null)
        //        throw new ArgumentNullException(nameof(inputKeys));
        //    if (inputValues == null)
        //        throw new ArgumentNullException(nameof(inputValues));
        //    if (configKeys == null)
        //        throw new ArgumentNullException(nameof(configKeys));
        //    if (configValues == null)
        //        throw new ArgumentNullException(nameof(configValues));

        //    var creator = OpMap.GetSymbolCreator(operatorName);
        //    NativeMethods.MXSymbolCreateAtomicSymbol(creator, 
        //                                             (uint)configKeys.Count,
        //                                             configKeys.ToArray(),
        //                                             configValues.ToArray(),
        //                                             out var handle);

        //    NativeMethods.MXSymbolCompose(handle, 
        //                                  operatorName,
        //                                  (uint)inputKeys.Count,
        //                                  inputKeys.ToArray(),
        //                                  inputValues.ToArray());

        //    blob_ptr_ = std::make_shared<SymBlob>(handle);
        //    this.NativePtr = @out;
        //}

        #endregion

        #region Properties

        public string Name
        {
            get
            {
                this.ThrowIfDisposed();
                if (this.NativePtr == IntPtr.Zero)
                    return null;

                NativeMethods.MXSymbolGetName(this.NativePtr, out var @out, out var success);
                if (@out == IntPtr.Zero)
                    return null;

                return Marshal.PtrToStringAnsi(@out);
            }
        }

        public Symbol this[int index]
        {
            get
            {
                this.ThrowIfDisposed();

                NativeMethods.MXSymbolGetOutput(this.NativePtr, (uint)index, out var @out);
                return new Symbol(@out);
            }
        }

        public Symbol this[string index]
        {
            get
            {
                this.ThrowIfDisposed();

                var outputs = this.ListOutputs();
                for (var i = 0; i < outputs.Count; i++)
                {
                    if (outputs[i] == index)
                        return this[i];
                }

                throw new KeyNotFoundException($"Cannot find output that matches name {index}");
            }
        }

        #endregion

        #region Methods

        public Executor Bind(Context context,
                             IList<NDArray> argArrays,
                             IList<NDArray> gradArrays,
                             IList<OpReqType> gradReqs,
                             IList<NDArray> auxArrays)
        {
            return new Executor(this,
                                context,
                                argArrays,
                                gradArrays,
                                gradReqs,
                                auxArrays,
                                new Dictionary<string, Context>());
        }

        public Executor Bind(Context context,
                             IList<NDArray> argArrays,
                             IList<NDArray> gradArrays,
                             IList<OpReqType> gradReqs,
                             IList<NDArray> auxArrays,
                             IDictionary<string, Context> groupToCtx)
        {
            return new Executor(this,
                                context,
                                argArrays,
                                gradArrays,
                                gradReqs,
                                auxArrays,
                                groupToCtx,
                                null);
        }

        public Executor Bind(Context context,
                             IList<NDArray> argArrays,
                             IList<NDArray> gradArrays,
                             IList<OpReqType> gradReqs,
                             IList<NDArray> auxArrays,
                             IDictionary<string, Context> groupToCtx,
                             Executor sharedExec)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (argArrays == null)
                throw new ArgumentNullException(nameof(argArrays));
            if (gradArrays == null)
                throw new ArgumentNullException(nameof(gradArrays));
            if (gradReqs == null)
                throw new ArgumentNullException(nameof(gradReqs));
            if (auxArrays == null)
                throw new ArgumentNullException(nameof(auxArrays));
            if (groupToCtx == null)
                throw new ArgumentNullException(nameof(groupToCtx));

            return new Executor(this,
                                context,
                                argArrays,
                                gradArrays,
                                gradReqs,
                                auxArrays,
                                groupToCtx,
                                sharedExec);
        }

        public IntPtr GetHandle()
        {
            this.ThrowIfDisposed();
            return this.NativePtr;
        }

        public static Symbol Group(IList<Symbol> symbols)
        {
            var handleList = symbols.Select(symbol => symbol.GetHandle()).ToArray();
            NativeMethods.MXSymbolCreateGroup((uint)handleList.Length, handleList, out var @out);
            return new Symbol(@out);
        }

        public void InferArgsMap(Context context,
                                 IDictionary<string, NDArray> argsMap,
                                 IDictionary<string, NDArray> knownArgs)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (argsMap == null)
                throw new ArgumentNullException(nameof(argsMap));
            if (knownArgs == null)
                throw new ArgumentNullException(nameof(knownArgs));

            this.ThrowIfDisposed();

            var argShapes = new Dictionary<string, IList<mx_uint>>();
            var inShapes = new List<List<mx_uint>>();
            var auxShapes = new List<List<mx_uint>>();
            var outShapes = new List<List<mx_uint>>();

            var argNameList = this.ListArguments();
            foreach (var argName in argNameList)
            {
                if (knownArgs.TryGetValue(argName, out var value))
                    argShapes[argName] = value.GetShape();
            }

            this.InferShape(argShapes, inShapes, auxShapes, outShapes);

            for (var i = 0; i < inShapes.Count; ++i)
            {
                var shape = inShapes[i];
                var argName = argNameList[i];
                if (knownArgs.TryGetValue(argName, out var value))
                {
                    argsMap[argName] = value;
                }
                else
                {
                    var array = new NDArray(shape.ToArray(), false);
                    argsMap[argName] = array;
                    NDArray.SampleGaussian(0, 1, array);
                }
            }
        }

        public void InferShape(IDictionary<String, IList<mx_uint>> argShapes,
                               IList<List<mx_uint>> inShape,
                               IList<List<mx_uint>> auxShape,
                               IList<List<mx_uint>> outShape)
        {
            if (argShapes == null)
                throw new ArgumentNullException(nameof(argShapes));
            if (inShape == null)
                throw new ArgumentNullException(nameof(inShape));
            if (auxShape == null)
                throw new ArgumentNullException(nameof(auxShape));
            if (outShape == null)
                throw new ArgumentNullException(nameof(outShape));

            this.ThrowIfDisposed();

            var argIndPtr = new List<mx_uint>();
            var argShapeData = new List<mx_uint>();

            foreach (var item in argShapes.Values)
            {
                argIndPtr.Add((uint)argShapeData.Count);
                foreach (var i in item)
                    argShapeData.Add(i);
            }

            argIndPtr.Add((uint)argShapeData.Count);

            unsafe
            {
                var keys = argShapes.Keys.ToArray();
                var argIndPtrArray = argIndPtr.ToArray();
                var argShapeDataArray = argShapeData.ToArray();
                {
                    mx_uint inShapeSize;
                    mx_uint* inShapeNdim;
                    mx_uint** inShapeData;

                    Logging.CHECK_EQ(NativeMethods.MXSymbolInferShape(this.NativePtr,
                                                                      (uint)argShapes.Count,
                                                                      keys,
                                                                      argIndPtrArray,
                                                                      argShapeDataArray,
                                                                      &inShapeSize,
                                                                      &inShapeNdim,
                                                                      &inShapeData,
                                                                      out var outShapeSize,
                                                                      out var outShapeNdim,
                                                                      out var outShapeData,
                                                                      out var auxShapeSize,
                                                                      out var auxShapeNdim,
                                                                      out var auxShapeData,
                                                                      out var complete), NativeMethods.OK);

                    if (complete == 0)
                        return;

                    for (var i = 0; i < inShapeSize; ++i)
                    {
                        inShape.Add(new List<mx_uint>());
                        for (var j = 0; j < inShapeNdim[i]; ++j)
                            inShape[i].Add(inShapeData[i][j]);
                    }

                    for (var i = 0; i < auxShapeSize; ++i)
                    {
                        auxShape.Add(new List<mx_uint>());
                        for (var j = 0; j < auxShapeNdim[i]; ++j)
                            auxShape[i].Add(auxShapeData[i][j]);
                    }

                    for (var i = 0; i < outShapeSize; ++i)
                    {
                        outShape.Add(new List<mx_uint>());
                        for (var j = 0; j < outShapeNdim[i]; ++j)
                            outShape[i].Add(outShapeData[i][j]);
                    }
                }
            }
        }

        public void InferExecutorArrays(Context context,
                                        IList<NDArray> argArrays,
                                        IList<NDArray> gradArrays,
                                        IList<OpReqType> gradReqs,
                                        IList<NDArray> auxArrays,
                                        IDictionary<string, NDArray> argsMap)
        {
            this.InferExecutorArrays(context,
                                     argArrays,
                                     gradArrays,
                                     gradReqs,
                                     auxArrays,
                                     argsMap,
                                     new Dictionary<string, NDArray>());
        }

        public void InferExecutorArrays(Context context,
                                        IList<NDArray> argArrays,
                                        IList<NDArray> gradArrays,
                                        IList<OpReqType> gradReqs,
                                        IList<NDArray> auxArrays,
                                        IDictionary<string, NDArray> argsMap,
                                        IDictionary<string, NDArray> argGradStore)
        {
            this.InferExecutorArrays(context,
                                     argArrays,
                                     gradArrays,
                                     gradReqs,
                                     auxArrays,
                                     argsMap,
                                     argGradStore,
                                     new Dictionary<string, OpReqType>());
        }

        public void InferExecutorArrays(Context context,
                                        IList<NDArray> argArrays,
                                        IList<NDArray> gradArrays,
                                        IList<OpReqType> gradReqs,
                                        IList<NDArray> auxArrays,
                                        IDictionary<string, NDArray> argsMap,
                                        IDictionary<string, NDArray> argGradStore,
                                        IDictionary<string, OpReqType> gradReqType)
        {
            this.InferExecutorArrays(context,
                                     argArrays,
                                     gradArrays,
                                     gradReqs,
                                     auxArrays,
                                     argsMap,
                                     argGradStore,
                                     gradReqType,
                                     new Dictionary<string, NDArray>());
        }

        public void InferExecutorArrays(Context context,
                                    IList<NDArray> argArrays,
                                    IList<NDArray> gradArrays,
                                    IList<OpReqType> gradReqs,
                                    IList<NDArray> auxArrays,
                                    IDictionary<string, NDArray> argsMap,
                                    IDictionary<string, NDArray> argGradStore,
                                    IDictionary<string, OpReqType> gradReqType,
                                    IDictionary<string, NDArray> auxMap)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (argArrays == null)
                throw new ArgumentNullException(nameof(argArrays));
            if (gradArrays == null)
                throw new ArgumentNullException(nameof(gradArrays));
            if (gradReqs == null)
                throw new ArgumentNullException(nameof(gradReqs));
            if (auxArrays == null)
                throw new ArgumentNullException(nameof(auxArrays));
            if (argsMap == null)
                throw new ArgumentNullException(nameof(argsMap));
            if (argGradStore == null)
                throw new ArgumentNullException(nameof(argGradStore));
            if (gradReqType == null)
                throw new ArgumentNullException(nameof(gradReqType));
            if (auxMap == null)
                throw new ArgumentNullException(nameof(auxMap));

            this.ThrowIfDisposed();

            var argNameList = this.ListArguments();
            var inShapes = new List<List<mx_uint>>();
            var auxShapes = new List<List<mx_uint>>();
            var outShapes = new List<List<mx_uint>>();
            var argShapes = new Dictionary<string, IList<mx_uint>>();

            foreach (var argName in argNameList)
            {
                if (argsMap.TryGetValue(argName, out var value))
                    argShapes[argName] = value.GetShape();
            }

            this.InferShape(argShapes, inShapes, auxShapes, outShapes);

            for (var i = 0; i < inShapes.Count; ++i)
            {
                var shape = inShapes[i];
                var argName = argNameList[i];
                if (argsMap.TryGetValue(argName, out var value1))
                {
                    argArrays.Add(value1);
                }
                else
                {
                    argArrays.Add(new NDArray(shape, false));
                    NDArray.SampleGaussian(0, 1, argArrays.Last());
                }

                if (argGradStore.TryGetValue(argName, out var value2))
                {
                    gradArrays.Add(value2);
                }
                else
                {
                    gradArrays.Add(new NDArray(shape, false));
                }

                if (gradReqType.TryGetValue(argName, out var value3))
                {
                    gradReqs.Add(value3);
                }
                else if (argName.LastIndexOf("data", StringComparison.InvariantCulture) == argName.Length - 4 ||
                         argName.LastIndexOf("label", StringComparison.InvariantCulture) == argName.Length - 5)
                {
                    gradReqs.Add(OpReqType.NullOp);
                }
                else
                {
                    gradReqs.Add(OpReqType.WriteTo);
                }
            }

            var auxNameList = this.ListAuxiliaryStates();
            for (var i = 0; i < auxShapes.Count; ++i)
            {
                var shape = auxShapes[i];
                var auxName = auxNameList[i];
                if (auxMap.TryGetValue(auxName, out var value))
                {
                    auxArrays.Add(value);
                }
                else
                {
                    auxArrays.Add(new NDArray(shape, false));
                    NDArray.SampleGaussian(0, 1, auxArrays.Last());
                }
            }
        }

        public IList<string> ListArguments()
        {
            this.ThrowIfDisposed();

            NativeMethods.MXSymbolListArguments(this.GetHandle(), out var size, out var sarry);
            var sarryArray = InteropHelper.ToPointerArray(sarry, size);

            var ret = new string[size];
            for (var i = 0; i < size; i++)
                ret[i] = Marshal.PtrToStringAnsi(sarryArray[i]);

            return ret;
        }

        public IList<string> ListAuxiliaryStates()
        {
            this.ThrowIfDisposed();

            NativeMethods.MXSymbolListAuxiliaryStates(this.GetHandle(), out var size, out var sarry);
            var sarryArray = InteropHelper.ToPointerArray(sarry, size);

            var ret = new string[size];
            for (var i = 0; i < size; i++)
                ret[i] = Marshal.PtrToStringAnsi(sarryArray[i]);

            return ret;
        }

        public IList<string> ListOutputs()
        {
            this.ThrowIfDisposed();

            NativeMethods.MXSymbolListOutputs(this.GetHandle(), out var size, out var sarry);
            var sarryArray = InteropHelper.ToPointerArray(sarry, size);
            var ret = new string[size];
            for (var i = 0; i < size; i++)
                ret[i] = Marshal.PtrToStringAnsi(sarryArray[i]);

            return ret;
        }

        public static Symbol Load(string fileName)
        {
            Logging.CHECK_EQ(NativeMethods.MXSymbolCreateFromFile(fileName, out var handle), NativeMethods.OK);
            return new Symbol(handle);
        }

        public static Symbol LoadJSON(string json)
        {
            Logging.CHECK_EQ(NativeMethods.MXSymbolCreateFromJSON(json, out var handle), NativeMethods.OK);
            return new Symbol(handle);
        }

        public void Save(string fileName)
        {
            Logging.CHECK_EQ(NativeMethods.MXSymbolSaveToFile(this.GetHandle(), fileName), NativeMethods.OK);
        }

        public Executor SimpleBind(Context context,
                                   IDictionary<string, NDArray> argsMap)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (argsMap == null)
                throw new ArgumentNullException(nameof(argsMap));

            this.ThrowIfDisposed();

            return this.SimpleBind(context, argsMap, new Dictionary<string, NDArray>());
        }

        public Executor SimpleBind(Context context,
                                   IDictionary<string, NDArray> argsMap,
                                   IDictionary<string, NDArray> argGradStore)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (argsMap == null)
                throw new ArgumentNullException(nameof(argsMap));
            if (argGradStore == null)
                throw new ArgumentNullException(nameof(argGradStore));

            this.ThrowIfDisposed();

            return this.SimpleBind(context, argsMap, argGradStore, new Dictionary<string, OpReqType>());
        }

        public Executor SimpleBind(Context context,
                                   IDictionary<string, NDArray> argsMap,
                                   IDictionary<string, NDArray> argGradStore,
                                   IDictionary<string, OpReqType> gradReqType)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (argsMap == null)
                throw new ArgumentNullException(nameof(argsMap));
            if (argGradStore == null)
                throw new ArgumentNullException(nameof(argGradStore));
            if (gradReqType == null)
                throw new ArgumentNullException(nameof(gradReqType));

            this.ThrowIfDisposed();

            return this.SimpleBind(context, argsMap, argGradStore, gradReqType, new Dictionary<string, NDArray>());
        }

        public Executor SimpleBind(Context context,
                                   IDictionary<string, NDArray> argsMap,
                                   IDictionary<string, NDArray> argGradStore,
                                   IDictionary<string, OpReqType> gradReqType,
                                   IDictionary<string, NDArray> auxMap)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (argsMap == null)
                throw new ArgumentNullException(nameof(argsMap));
            if (argGradStore == null)
                throw new ArgumentNullException(nameof(argGradStore));
            if (gradReqType == null)
                throw new ArgumentNullException(nameof(gradReqType));
            if (auxMap == null)
                throw new ArgumentNullException(nameof(auxMap));

            this.ThrowIfDisposed();

            var argArrays = new List<NDArray>();
            var gradArrays = new List<NDArray>();
            var gradReqs = new List<OpReqType>();
            var auxArrays = new List<NDArray>();

            this.InferExecutorArrays(context,
                                     argArrays,
                                     gradArrays,
                                     gradReqs,
                                     auxArrays,
                                     argsMap,
                                     argGradStore,
                                     gradReqType,
                                     auxMap);

            return new Executor(this, context, argArrays, gradArrays, gradReqs, auxArrays);
        }

        public string ToJSON()
        {
            Logging.CHECK_EQ(NativeMethods.MXSymbolSaveToJSON(this.GetHandle(), out var outJson), NativeMethods.OK);
            return Marshal.PtrToStringAnsi(outJson);
        }

        public static Symbol Variable(string name)
        {
            return new Symbol(name);
        }

        #region Overrides

        #region Operators

        public static Symbol operator +(Symbol lhs, Symbol rhs)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            lhs.ThrowIfDisposed();
            rhs.ThrowIfDisposed();

            return OperatorSupply.Plus(lhs, rhs);
        }

        public static Symbol operator -(Symbol lhs, Symbol rhs)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            lhs.ThrowIfDisposed();
            rhs.ThrowIfDisposed();

            return OperatorSupply.Minus(lhs, rhs);
        }

        public static Symbol operator *(Symbol lhs, Symbol rhs)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            lhs.ThrowIfDisposed();
            rhs.ThrowIfDisposed();

            return OperatorSupply.Mul(lhs, rhs);
        }

        public static Symbol operator /(Symbol lhs, Symbol rhs)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            lhs.ThrowIfDisposed();
            rhs.ThrowIfDisposed();

            return OperatorSupply.Div(lhs, rhs);
        }

        public static Symbol operator %(Symbol lhs, Symbol rhs)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            lhs.ThrowIfDisposed();
            rhs.ThrowIfDisposed();

            return OperatorSupply.Mod(lhs, rhs);
        }

        public static Symbol operator +(Symbol lhs, float scalar)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));

            lhs.ThrowIfDisposed();

            return OperatorSupply.PlusScalar(lhs, scalar);
        }

        public static Symbol operator +(float lhs, Symbol rhs)
        {
            return rhs + lhs;
        }

        public static Symbol operator -(Symbol lhs, float scalar)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));

            lhs.ThrowIfDisposed();

            return OperatorSupply.MinimumScalar(lhs, scalar);
        }

        public static Symbol operator -(float lhs, Symbol rhs)
        {
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            rhs.ThrowIfDisposed();

            return OperatorSupply.RMinusScalar(lhs, rhs);
        }

        public static Symbol operator *(Symbol lhs, float scalar)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));

            lhs.ThrowIfDisposed();

            return OperatorSupply.MulScalar(lhs, scalar);
        }

        public static Symbol operator *(float lhs, Symbol rhs)
        {
            return rhs * lhs;
        }

        public static Symbol operator /(Symbol lhs, float scalar)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));

            lhs.ThrowIfDisposed();

            return OperatorSupply.DivScalar(lhs, scalar);
        }

        public static Symbol operator /(float lhs, Symbol rhs)
        {
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            rhs.ThrowIfDisposed();

            return OperatorSupply.RDivScalar(lhs, rhs);
        }

        public static Symbol operator %(Symbol lhs, float scalar)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));

            lhs.ThrowIfDisposed();

            return OperatorSupply.ModScalar(lhs, scalar);
        }

        public static Symbol operator %(float lhs, Symbol rhs)
        {
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            rhs.ThrowIfDisposed();

            return OperatorSupply.RModScalar(lhs, rhs);
        }

        #endregion

        #endregion

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
