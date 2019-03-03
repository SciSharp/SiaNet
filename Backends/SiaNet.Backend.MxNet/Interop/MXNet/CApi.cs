using System.Runtime.InteropServices;
using AtomicSymbolCreator = System.IntPtr;
using DataIterCreator = System.IntPtr;
using DataIterHandle = System.IntPtr;
using ExecutorHandle = System.IntPtr;
using NDArrayHandle = System.IntPtr;
using SymbolHandle = System.IntPtr;
using size_t = System.UInt64;
using uint64_t = System.UInt64;
using mx_uint = System.UInt32;
using mx_float = System.Single;

using ExecutorMonitorCallback = System.IntPtr;
using System;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib.Interop
{

    internal sealed partial class NativeMethods
    {

        #region Callbacks

        public delegate void ExecutorMonitorCallbackDelegate(string str, NDArrayHandle arrayHandle, ExecutorHandle executeHandle);

        #endregion

        #region Methods

        #region Part 0: Global State setups

        /// <summary>
        /// Notify the engine about a shutdown, This can help engine to print less messages into display.
        /// <para>User do not have to call this function.</para>
        /// </summary>
        /// <returns>0 when success, -1 when failure happens.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNotifyShutdown();

        #endregion

        #region Part 1: NDArray creation and deletion

        /// <summary>
        /// create a NDArray handle that is not initialized can be used to pass in as mutate variables to hold the result of NDArray
        /// </summary>
        /// <param name="out">the returning handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayCreateNone(out NDArrayHandle @out);

        /// <summary>
        /// free the narray handle
        /// </summary>
        /// <param name="symbol">the handle to be freed</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayFree(NDArrayHandle symbol);

        /// <summary>
        /// get the context of the NDArray
        /// </summary>
        /// <param name="handle">the handle to the narray</param>
        /// <param name="out_dev_type">the output device type</param>
        /// <param name="out_dev_id">the output device id</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetContext(NDArrayHandle handle,
                                                     out int out_dev_type,
                                                     out int out_dev_id);

        /// <summary>
        /// get the content of the data in NDArray
        /// </summary>
        /// <param name="handle">the handle to the ndarray</param>
        /// <param name="out_pdata">pointer holder to get pointer of data</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetData(NDArrayHandle handle, out AtomicSymbolCreator out_pdata);

        /// <summary>
        /// get the content of the data in NDArray
        /// </summary>
        /// <param name="handle">the handle to the ndarray</param>
        /// <param name="out_dtype">pointer holder to get pointer of data</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetDType(NDArrayHandle handle, out int out_dtype);

        /// <summary>
        /// Load list of narray from the file.
        /// </summary>
        /// <param name="fname">name of the file.</param>
        /// <param name="out_size">number of narray loaded.</param>
        /// <param name="out_arr">head of the returning narray handles.</param>
        /// <param name="out_name_size">size of output name arrray.</param>
        /// <param name="out_names">the names of returning NDArrays, can be NULL</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayLoad([MarshalAs(UnmanagedType.LPStr)] string fname,
                                               out mx_uint out_size,
                                               out System.IntPtr out_arr,
                                               out mx_uint out_name_size,
                                               out System.IntPtr out_names);

        /// <summary>
        /// Save list of narray into the file.
        /// </summary>
        /// <param name="fname">name of the file.</param>
        /// <param name="num_args">number of arguments to save.</param>
        /// <param name="args">the array of NDArrayHandles to be saved.</param>
        /// <param name="keys">the name of the NDArray, optional, can be NULL</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySave([MarshalAs(UnmanagedType.LPStr)]string fname,
                                               mx_uint num_args,
                                               NDArrayHandle[] args,
                                               [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys);

        /// <summary>
        /// get the shape of the array
        /// </summary>
        /// <param name="handle">the handle to the narray</param>
        /// <param name="out_dim">the output dimension</param>
        /// <param name="out_pdata">pointer holder to get data pointer of the shape</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetShape(NDArrayHandle handle,
                                                   out mx_uint out_dim,
                                                   out AtomicSymbolCreator out_pdata);

        /// <summary>
        /// Slice the NDArray along axis 0.
        /// </summary>
        /// <param name="handle">the handle to the NDArray</param>
        /// <param name="slice_begin">The beginning index of slice</param>
        /// <param name="slice_end">The ending index of slice</param>
        /// <param name="out">The NDArrayHandle of sliced NDArray</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySlice(NDArrayHandle handle,
                                                mx_uint slice_begin,
                                                mx_uint slice_end,
                                                out NDArrayHandle @out);

        /// <summary>
        /// Perform a synchronize copy from a continugous CPU memory region.
        /// <para>This function will call WaitToWrite before the copy is performed. This is useful to copy data from existing memory region that are not wrapped by NDArray(thus dependency not being tracked).</para>
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <param name="data">the data source to copy from.</param>
        /// <param name="size">the memory size we want to copy from.</param>
        /// <returns></returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySyncCopyFromCPU(NDArrayHandle handle,
                                                          mx_float[] data,
                                                          size_t size);

        /// <summary>
        /// Perform a synchronize copyto a continugous CPU memory region.
        /// <para>This function will call WaitToRead before the copy is performed. This is useful to copy data from existing memory region that are not wrapped by NDArray(thus dependency not being tracked).</para>
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <param name="data">the data source to copy into.</param>
        /// <param name="size">the memory size we want to copy into.</param>
        /// <returns></returns>
        //[DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        //public static extern int MXNDArraySyncCopyToCPU(NDArrayHandle handle,
        //                                                mx_float[] data,
        //                                                size_t size);

        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///data: void*
        ///size: size_t->unsigned int
        [DllImport(NativeLibrary, EntryPoint = "MXNDArraySyncCopyToCPU", CallingConvention = CallingConvention)]
        public static extern int MXNDArraySyncCopyToCPU(IntPtr handle, IntPtr data, size_t size);

        /// <summary>
        /// wait until all delayed operations in the system is completed
        /// </summary>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayWaitAll();

        /// <summary>
        /// Wait until all the pending writes with respect NDArray are finished. Always call this before read data out synchronizely.
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayWaitToRead(NDArrayHandle handle);

        /// <summary>
        /// Wait until all the pending read/write with respect NDArray are finished. Always call this before write data into NDArray synchronizely.
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayWaitToWrite(NDArrayHandle handle);

        #endregion

        #region Part 2: functions on NDArray

        /// <summary>
        /// invoke a nnvm op and imperative function
        /// </summary>
        /// <param name="creator">the op</param>
        /// <param name="num_inputs">number of input NDArrays</param>
        /// <param name="inputs">input NDArrays</param>
        /// <param name="num_outputs">number of output NDArrays</param>
        /// <param name="outputs">output NDArrays</param>
        /// <param name="num_params">number of keyword parameters</param>
        /// <param name="param_keys">keys for keyword parameters</param>
        /// <param name="param_vals">values for keyword parameters</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXImperativeInvoke(AtomicSymbolCreator creator,
                                                    int num_inputs,
                                                    NDArrayHandle[] inputs,
                                                    ref int num_outputs,
                                                    ref NDArrayHandle[] outputs,
                                                    int num_params,
                                                    [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] param_keys,
                                                    [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] param_vals);

        #endregion

        #region Part 3: symbolic configuration generation

        /// <summary>
        /// list all the available operator names, include entries.
        /// </summary>
        /// <param name="out_size">the size of returned array</param>
        /// <param name="out_array">the output operator name array.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXListAllOpNames(out mx_uint out_size, out AtomicSymbolCreator[] out_array);

        /// <summary>
        /// This function will change the sym hanlde. To achieve function apply behavior, copy the symbol first before apply.
        /// </summary>
        /// <param name="sym">the symbol to apply</param>
        /// <param name="name">the name of symbol</param>
        /// <param name="num_args">number of arguments</param>
        /// <param name="keys">the key of keyword args(optional)</param>
        /// <param name="args">arguments to sym</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCompose(SymbolHandle sym,
                                                 [MarshalAs(UnmanagedType.LPStr)] string name,
                                                 mx_uint num_args,
                                                 [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
                                                 SymbolHandle[] args);

        /// <summary>
        /// Create an AtomicSymbol.
        /// </summary>
        /// <param name="creator">the AtomicSymbolCreator</param>
        /// <param name="num_param"> the number of parameters</param>
        /// <param name="keys">keys to the params</param>
        /// <param name="vals">the vals of the params</param>
        /// <param name="@out">pointer to the created symbol handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                                                            mx_uint num_param,
                                                            [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
                                                            [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] vals,
                                                            out SymbolHandle @out);

        /// <summary>
        /// Load a symbol from a json file.
        /// </summary>
        /// <param name="fname">the file name.</param>
        /// <param name="out">the output symbol.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateFromFile([MarshalAs(UnmanagedType.LPStr)] string fname, out SymbolHandle @out);

        /// <summary>
        /// Load a symbol from a json string.
        /// </summary>
        /// <param name="json">the json string.</param>
        /// <param name="out">the output symbol.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateFromJSON([MarshalAs(UnmanagedType.LPStr)] string json, out SymbolHandle @out);

        /// <summary>
        /// Create a Symbol by grouping list of symbols together
        /// </summary>
        /// <param name="num_symbols">number of symbols to be grouped</param>
        /// <param name="symbols">array of symbol handles</param>
        /// <param name="@out">pointer to the created symbol handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateGroup(mx_uint num_symbols,
                                                     SymbolHandle[] symbols,
                                                     out SymbolHandle @out);

        /// <summary>
        /// Create a Variable Symbol.
        /// </summary>
        /// <param name="name">name of the variable</param>
        /// <param name="@out">pointer to the created symbol handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateVariable([MarshalAs(UnmanagedType.LPStr)] string name, out SymbolHandle @out);

        /// <summary>
        /// Free the symbol handle.
        /// </summary>
        /// <param name="symbol">symbol the symbol</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolFree(SymbolHandle symbol);

        /// <summary>
        /// Get string name from symbol
        /// </summary>
        /// <param name="symbol">the source symbol</param>
        /// <param name="out">The result name.</param>
        /// <param name="success">Whether the result is contained in out.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetName(SymbolHandle symbol, out System.IntPtr @out, out int success);

        /// <summary>
        /// Get index-th outputs of the symbol.
        /// </summary>
        /// <param name="symbol">The symbol</param>
        /// <param name="index">the Index of the output.</param>
        /// <param name="@out">The output symbol whose outputs are the index-th symbol.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetOutput(SymbolHandle symbol,
                                                   mx_uint index,
                                                   out SymbolHandle @out);

        /// <summary>
        /// Get the detailed information about atomic symbol.
        /// </summary>
        /// <param name="creator">the AtomicSymbolCreator.</param>
        /// <param name="name">The returned name of the creator.</param>
        /// <param name="description">The returned description of the symbol.</param>
        /// <param name="num_args">Number of arguments.</param>
        /// <param name="arg_names">Name of the arguments.</param>
        /// <param name="arg_type_infos">Type informations about the arguments.</param>
        /// <param name="arg_descriptions">Description information about the arguments.</param>
        /// <param name="key_var_num_args">
        /// The keyword argument for specifying variable number of arguments.
        /// <para>When this parameter has non-zero length, the function allows variable number of positional arguments, and will need the caller to pass it in in MXSymbolCreateAtomicSymbol, With key = key_var_num_args, and value = number of positional arguments.</para>
        /// </param>
        /// <param name="return_type">Return type of the function, can be Symbol or Symbol[]</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                                             out AtomicSymbolCreator name,
                                                             out AtomicSymbolCreator description,
                                                             out mx_uint num_args,
                                                             out AtomicSymbolCreator arg_names,
                                                             out AtomicSymbolCreator arg_type_infos,
                                                             out AtomicSymbolCreator arg_descriptions,
                                                             out AtomicSymbolCreator key_var_num_args,
                                                             ref AtomicSymbolCreator return_type);

        /// <summary>
        /// Get the detailed information about atomic symbol.
        /// </summary>
        /// <param name="creator">the AtomicSymbolCreator.</param>
        /// <param name="name">The returned name of the creator.</param>
        /// <param name="description">The returned description of the symbol.</param>
        /// <param name="num_args">Number of arguments.</param>
        /// <param name="arg_names">Name of the arguments.</param>
        /// <param name="arg_type_infos">Type informations about the arguments.</param>
        /// <param name="arg_descriptions">Description information about the arguments.</param>
        /// <param name="key_var_num_args">
        /// The keyword argument for specifying variable number of arguments.
        /// <para>When this parameter has non-zero length, the function allows variable number of positional arguments, and will need the caller to pass it in in MXSymbolCreateAtomicSymbol, With key = key_var_num_args, and value = number of positional arguments.</para>
        /// </param>
        /// <param name="return_type">Return type of the function, can be Symbol or Symbol[]</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                                             out AtomicSymbolCreator name,
                                                             out AtomicSymbolCreator description,
                                                             out mx_uint num_args,
                                                             out AtomicSymbolCreator[] arg_names,
                                                             out AtomicSymbolCreator[] arg_type_infos,
                                                             out AtomicSymbolCreator[] arg_descriptions,
                                                             out AtomicSymbolCreator key_var_num_args,
                                                             out AtomicSymbolCreator return_type);

        /// <summary>
        /// infer shape of unknown input shapes given the known one.
        /// <para>The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data</para>
        /// <para>The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.</para>
        /// </summary>
        /// <param name="sym">symbol handle</param>
        /// <param name="num_args">numbe of input arguments.</param>
        /// <param name="keys">the key of keyword args (optional)</param>
        /// <param name="arg_ind_ptr">the head pointer of the rows in CSR</param>
        /// <param name="arg_shape_data">the content of the CSR</param>
        /// <param name="in_shape_size">sizeof the returning array of in_shapes</param>
        /// <param name="in_shape_ndim">returning array of shape dimensions of eachs input shape.</param>
        /// <param name="in_shape_data">returning array of pointers to head of the input shape.</param>
        /// <param name="out_shape_size">sizeof the returning array of out_shapes</param>
        /// <param name="out_shape_ndim">returning array of shape dimensions of eachs input shape.</param>
        /// <param name="out_shape_data">returning array of pointers to head of the input shape.</param>
        /// <param name="aux_shape_size">sizeof the returning array of aux_shapes</param>
        /// <param name="aux_shape_ndim">returning array of shape dimensions of eachs auxiliary shape.</param>
        /// <param name="aux_shape_data">returning array of pointers to head of the auxiliary shape.</param>
        /// <param name="complete">whether infer shape completes or more information is needed.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern unsafe int MXSymbolInferShape(SymbolHandle sym,
                                                           mx_uint num_args,
                                                           [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
                                                           mx_uint[] arg_ind_ptr,
                                                           mx_uint[] arg_shape_data,
                                                           mx_uint* in_shape_size,
                                                           mx_uint** in_shape_ndim,
                                                           mx_uint*** in_shape_data,
                                                           out mx_uint out_shape_size,
                                                           out mx_uint* out_shape_ndim,
                                                           out mx_uint** out_shape_data,
                                                           out mx_uint aux_shape_size,
                                                           out mx_uint* aux_shape_ndim,
                                                           out mx_uint** aux_shape_data,
                                                           out int complete);

        /// <summary>
        /// List arguments in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_size">output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListArguments(SymbolHandle symbol,
                                                       out mx_uint out_size,
                                                       out AtomicSymbolCreator out_str_array);

        /// <summary>
        /// list all the available AtomicSymbolEntry
        /// </summary>
        /// <param name="out_size">the size of returned array</param>
        /// <param name="out_array">the output AtomicSymbolCreator array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListAtomicSymbolCreators(out mx_uint out_size, out AtomicSymbolCreator out_array);

        /// <summary>
        /// List auxiliary states in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_size">output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                                             out mx_uint out_size,
                                                             out AtomicSymbolCreator out_str_array);

        /// <summary>
        /// List returns in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_size">output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListOutputs(SymbolHandle symbol,
                                                     out mx_uint out_size,
                                                     out AtomicSymbolCreator out_str_array);

        /// <summary>
        /// Save a symbol into a json file.
        /// </summary>
        /// <param name="symbol">the input symbol.</param>
        /// <param name="fname">the file name.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolSaveToFile(SymbolHandle symbol, [MarshalAs(UnmanagedType.LPStr)] string fname);

        /// <summary>
        /// Save a symbol into a json string
        /// </summary>
        /// <param name="symbol">the input symbol.</param>
        /// <param name="out_json">output json string.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolSaveToJSON(SymbolHandle symbol, out System.IntPtr out_json);

        #endregion

        #region Part 4: Executor interface

        /// <summary>
        /// Excecutor run backward
        /// </summary>
        /// <param name="handle">execute handle</param>
        /// <param name="len">lenth</param>
        /// <param name="head_grads">NDArray handle for heads' gradient</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorBackward(ExecutorHandle handle, mx_uint len, NDArrayHandle[] head_grads);

        /*!
         * \brief Generate Executor from symbol,
         *  This is advanced function, allow specify group2ctx map.
         *  The user can annotate "ctx_group" attribute to name each group.
         *
         * \param symbol_handle symbol handle
         * \param dev_type device type of default context
         * \param dev_id device id of default context
         * \param num_map_keys size of group2ctx map
         * \param map_keys keys of group2ctx map
         * \param map_dev_types device type of group2ctx map
         * \param map_dev_ids device id of group2ctx map
         * \param len length
         * \param in_args in args array
         * \param arg_grad_store arg grads handle array
         * \param grad_req_type grad req array
         * \param aux_states_len length of auxiliary states
         * \param aux_states auxiliary states array
         * \param shared_exec input executor handle for memory sharing
         * \param out output executor handle
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorBindEX(SymbolHandle symbol_handle,
                                                  int dev_type,
                                                  int dev_id,
                                                  mx_uint num_map_keys,
                                                  [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] map_keys,
                                                  int[] map_dev_types,
                                                  int[] map_dev_ids,
                                                  mx_uint len,
                                                  NDArrayHandle[] in_args,
                                                  NDArrayHandle[] arg_grad_store,
                                                  mx_uint[] grad_req_type,
                                                  mx_uint aux_states_len,
                                                  NDArrayHandle[] aux_states,
                                                  ExecutorHandle shared_exec,
                                                  out ExecutorHandle @out);

        /// <summary>
        /// Executor forward method
        /// </summary>
        /// <param name="handle">executor handle</param>
        /// <param name="is_train">int value to indicate whether the forward pass is for evaluation</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorForward(ExecutorHandle handle, int is_train);

        /// <summary>
        /// Delete the executor
        /// </summary>
        /// <param name="handle">the executor.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorFree(ExecutorHandle handle);

        /// <summary>
        /// Get executor's head NDArray
        /// </summary>
        /// <param name="handle">executor handle</param>
        /// <param name="out_size">output narray vector size</param>
        /// <param name="out">out put narray handles</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorOutputs(ExecutorHandle handle, out mx_uint out_size, out AtomicSymbolCreator @out);

        /// <summary>
        /// Print the content of execution plan, used for debug.
        /// </summary>
        /// <param name="handle">the executor.</param>
        /// <param name="out_str">pointer to hold the output string of the printing.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorPrint(ExecutorHandle handle, out AtomicSymbolCreator out_str);

        //[DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        //public static extern int MXExecutorSimpleBind(SymbolHandle symbol_handle,
        //                           int dev_type,
        //                           int dev_id,
        //                           const mx_uint num_g2c_keys,
        //                           const char** g2c_keys,
        //                           const int* g2c_dev_types,
        //                           const int* g2c_dev_ids,
        //                           const mx_uint provided_grad_req_list_len,
        //                           const char** provided_grad_req_names,
        //                           const char** provided_grad_req_types,
        //                           const mx_uint num_provided_arg_shapes,
        //                           const char** provided_arg_shape_names,
        //                           const mx_uint* provided_arg_shape_data,
        //                           const mx_uint* provided_arg_shape_idx,
        //                           const mx_uint num_provided_arg_dtypes,
        //                           const char** provided_arg_dtype_names,
        //                           const int* provided_arg_dtypes,
        //                           const mx_uint num_provided_arg_stypes,
        //                           const char** provided_arg_stype_names,
        //                           const int* provided_arg_stypes,
        //                           const mx_uint num_shared_arg_names,
        //                           const char** shared_arg_name_list,
        //                           int* shared_buffer_len,
        //                           const char** shared_buffer_name_list,
        //                           NDArrayHandle* shared_buffer_handle_list,
        //                           const char*** updated_shared_buffer_name_list,
        //                           NDArrayHandle** updated_shared_buffer_handle_list,
        //                           mx_uint* num_in_args,
        //                           NDArrayHandle** in_args,
        //                           NDArrayHandle** arg_grads,
        //                           mx_uint* num_aux_states,
        //                           NDArrayHandle** aux_states,
        //                           ExecutorHandle shared_exec_handle,
        //                           ExecutorHandle* out);

        #endregion

        #region Part 5: IO Interface

        /// <summary>
        /// set a call back to notify the completion of operation
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="callback"></param>
        /// <param name="callback_handle"></param>
        /// <returns></returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorSetMonitorCallback(ExecutorHandle handle,
                                                              ExecutorMonitorCallback callback,
                                                              AtomicSymbolCreator callback_handle);

        /// <summary>
        /// List all the available iterator entries
        /// </summary>
        /// <param name="out_size">the size of returned iterators</param>
        /// <param name="out_array">the output iteratos entries</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXListDataIters(out mx_uint out_size, out AtomicSymbolCreator out_array);

        /// <summary>
        /// Init an iterator, init with parameters the array size of passed in arguments
        /// </summary>
        /// <param name="handle">handle of the iterator creator</param>
        /// <param name="num_param">number of parameter</param>
        /// <param name="keys">parameter keys</param>
        /// <param name="vals">parameter values</param>
        /// <param name="out">resulting iterator</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterCreateIter(DataIterCreator handle,
                                                      mx_uint num_param,
                                                      [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
                                                      [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] vals,
                                                      out DataIterHandle @out);

        /// <summary>
        /// Get the detailed information about data iterator.
        /// </summary>
        /// <param name="creator">the DataIterCreator.</param>
        /// <param name="name">The returned name of the creator.</param>
        /// <param name="description">The returned description of the symbol.</param>
        /// <param name="num_args">Number of arguments.</param>
        /// <param name="arg_names">Name of the arguments.</param>
        /// <param name="arg_type_infos">Type informations about the arguments.</param>
        /// <param name="arg_descriptions">Description information about the arguments.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetIterInfo(DataIterCreator creator,
                                                       out AtomicSymbolCreator name,
                                                       out AtomicSymbolCreator description,
                                                       out mx_uint num_args,
                                                       out AtomicSymbolCreator arg_names,
                                                       out AtomicSymbolCreator arg_type_infos,
                                                       out AtomicSymbolCreator arg_descriptions);

        /// <summary>
        /// Free the handle to the IO module
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterFree(DataIterHandle handle);

        /// <summary>
        /// Move iterator to next position
        /// </summary>
        /// <param name="handle">the handle to iterator</param>
        /// <param name="out">return value of next</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterNext(DataIterHandle handle, out int @out);

        /// <summary>
        /// Call iterator.Reset
        /// </summary>
        /// <param name="handle">the handle to iterator</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterBeforeFirst(DataIterHandle handle);

        /// <summary>
        /// Get the handle to the NDArray of underlying data
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <param name="out">handle to underlying data NDArray</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetData(DataIterHandle handle, out NDArrayHandle @out);

        /// <summary>
        /// Get the image index by array.
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <param name="out_index">output index of the array.</param>
        /// <param name="out_size">output size of the array.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetIndex(DataIterHandle handle,
                                                    out AtomicSymbolCreator out_index,
                                                    out uint64_t out_size);

        /// <summary>
        /// Get the padding number in current data batch
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <param name="pad">pad number ptr</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetPadNum(DataIterHandle handle, out int pad);

        /// <summary>
        /// Get the handle to the NDArray of underlying label
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <param name="out">the handle to underlying label NDArray</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetLabel(DataIterHandle handle, out NDArrayHandle @out);

        #endregion

        #region Part 6: advanced KVStore for multi-machines

        /// <summary>
        /// create a NDArray with specified shape
        /// </summary>
        /// <param name="shape">the pointer to the shape</param>
        /// <param name="ndim">the dimension of the shape</param>
        /// <param name="dev_type">device type, specify device we want to take</param>
        /// <param name="dev_id">the device id of the specific device</param>
        /// <param name="delay_alloc">whether to delay allocation until the narray is first mutated</param>
        /// <param name="@out">the returning handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayCreate(mx_uint[] shape,
                                                 mx_uint ndim,
                                                 int dev_type,
                                                 int dev_id,
                                                 int delay_alloc,
                                                 out NDArrayHandle @out);


        #endregion

        #endregion

    }

}
