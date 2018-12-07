using System.Collections.Generic;
using System.Runtime.InteropServices;
using SiaNet.Backend.Interop;
using AtomicSymbolCreator = System.IntPtr;
using OpHandle = System.IntPtr;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    /// <summary>
    /// OpMap instance holds a map of all the symbol creators so we can get symbol creators by name. This is used internally by Symbol and Operator. This class cannot be inherited.
    /// </summary>
    public sealed class OpMap
    {

        #region Fields

        private readonly Dictionary<string, AtomicSymbolCreator> _SymbolCreators;

        private readonly Dictionary<string, OpHandle> _OpHandles;

        #endregion

        #region Constructors

        public OpMap()
        {
            var r = NativeMethods.MXSymbolListAtomicSymbolCreators(out var numSymbolCreators, out var symbolCreators);
            Logging.CHECK_EQ(r, NativeMethods.OK);

            this._SymbolCreators = new Dictionary<string, AtomicSymbolCreator>((int)numSymbolCreators);

            var symbolCreatorsArray = InteropHelper.ToPointerArray(symbolCreators, numSymbolCreators);
            for (var i = 0; i < numSymbolCreators; i++)
            {
                var return_type = System.IntPtr.Zero;
                r = NativeMethods.MXSymbolGetAtomicSymbolInfo(symbolCreatorsArray[i],
                                                              out var name,
                                                              out var description,
                                                              out var numArgs,
                                                              out var argNames,
                                                              out var argTypeInfos,
                                                              out var argDescriptions,
                                                              out var nameBuilder,
                                                              ref return_type);
                Logging.CHECK_EQ(r, NativeMethods.OK);
                var str = Marshal.PtrToStringAnsi(name);
                this._SymbolCreators.Add(str, symbolCreatorsArray[i]);
            }

            r = NativeMethods.NNListAllOpNames(out var numOps, out var opNames);
            Logging.CHECK_EQ(r, NativeMethods.OK);

            this._OpHandles = new Dictionary<string, AtomicSymbolCreator>((int)numOps);

            var opNamesArray = InteropHelper.ToPointerArray(opNames, numOps);
            for (var i = 0; i < numOps; i++)
            {
                r = NativeMethods.NNGetOpHandle(opNamesArray[i], out var handle);
                Logging.CHECK_EQ(r, NativeMethods.OK);
                var str = Marshal.PtrToStringAnsi(opNamesArray[i]);
                this._OpHandles.Add(str, handle);
            }
        }

        #endregion

        #region Methods

        public OpHandle GetOpHandle(string name)
        {
            return this._OpHandles[name];
        }

        public AtomicSymbolCreator GetSymbolCreator(string name)
        {
            if (!this._SymbolCreators.TryGetValue(name, out var handle))
                return GetOpHandle(name);

            return handle;
        }

        #endregion

    }

}
