using System;
using System.Runtime.InteropServices;
using SiaNet.Backend.Interop;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed partial class MXNet
    {

        #region Methods

        public static void MXNDListCreate(byte[] nd_file_bytes,
                                          int nd_file_size,
                                          out NDListHandle handle,
                                          out uint out_length)
        {
            var ret = NativeMethods.MXNDListCreate(nd_file_bytes, nd_file_size, out var @out, out out_length);
            if (ret != NativeMethods.OK)
                throw CreateMXNetException($"Failed to create {nameof(PredictorHandle)}");

            handle = new NDListHandle(@out);
        }

        public static void MXNDListGet(NDListHandle handle,
                                       uint index,
                                       out string out_key,
                                       out float[] out_data,
                                       out uint[] out_shape,
                                       out uint out_ndim)
        {
            var ret = NativeMethods.MXNDListGet(handle.NativePtr, index, out var out_key_ptr, out var out_data_ptr, out var out_shape_ptr, out out_ndim);
            if (ret != NativeMethods.OK)
                throw CreateMXNetException($"Failed to get list from {nameof(NDListHandle)}");

            out_shape = new uint[out_ndim];
            NativeMethods.memcpy(out_shape, out_shape_ptr, out_ndim * sizeof(uint));

            var size = 1u;
            for (var i = 0u; i < out_ndim; ++i) size *= out_shape[i];

            out_data = new float[size];
            NativeMethods.memcpy(out_data, out_data_ptr, size * sizeof(float));

            out_key = Marshal.PtrToStringAnsi(out_key_ptr);
        }

        public static void MXPredCreate(string symbolJson,
                                        byte[] bytes,
                                        int size,
                                        int type,
                                        int deviceId,
                                        uint inputNodes,
                                        string[] keys,
                                        uint[] inputShapeIndptr,
                                        uint[] inputShapeData,
                                        out PredictorHandle handle)
        {
            var ret = NativeMethods.MXPredCreate(symbolJson, bytes, size, type, deviceId, inputNodes, keys, inputShapeIndptr, inputShapeData, out var @out);
            if (ret != NativeMethods.OK)
                throw CreateMXNetException($"Failed to create {nameof(PredictorHandle)}");

            handle = new PredictorHandle(@out);
        }

        public static void MXPredCreatePartialOut(string symbolJson,
                                                  byte[] bytes,
                                                  int size,
                                                  int type,
                                                  int deviceId,
                                                  uint inputNodes,
                                                  string[] keys,
                                                  uint[] inputShapeIndptr,
                                                  uint[] inputShapeData,
                                                  uint num_output_nodes,
                                                  string[] output_keys,
                                                  out PredictorHandle handle)
        {
            var ret = NativeMethods.MXPredCreatePartialOut(symbolJson, bytes, size, type, deviceId, inputNodes, keys, inputShapeIndptr, inputShapeData, num_output_nodes, output_keys, out var @out);
            if (ret != NativeMethods.OK)
                throw CreateMXNetException($"Failed to create {nameof(PredictorHandle)}");

            handle = new PredictorHandle(@out);
        }

        public static void MXPredForward(PredictorHandle handle)
        {
            if (handle == null)
                throw new ArgumentNullException(nameof(handle));

            handle.ThrowIfDisposed();

            var ret = NativeMethods.MXPredForward(handle.NativePtr);
            if (ret != NativeMethods.OK)
                throw CreateMXNetException($"Failed to forward {nameof(handle)}");
        }

        public static void MXPredGetOutput(PredictorHandle handle, uint index, float[] data, uint size)
        {
            if (handle == null)
                throw new ArgumentNullException(nameof(handle));

            handle.ThrowIfDisposed();

            var ret = NativeMethods.MXPredGetOutput(handle.NativePtr, index, data, size);
            if (ret != NativeMethods.OK)
                throw CreateMXNetException($"Failed to get output from {nameof(handle)}");
        }

        public static void MXPredGetOutputShape(PredictorHandle handle, uint index, out uint[] shape_data, out uint shape_ndim)
        {
            if (handle == null)
                throw new ArgumentNullException(nameof(handle));

            handle.ThrowIfDisposed();

            var ret = NativeMethods.MXPredGetOutputShape(handle.NativePtr, index, out var shape_data_ptr, out shape_ndim);
            if (ret != NativeMethods.OK)
                throw CreateMXNetException($"Failed to get output shape from {nameof(handle)}");

            shape_data = new uint[shape_ndim];
            NativeMethods.memcpy(shape_data, shape_data_ptr, shape_ndim * sizeof(uint));
        }
        
        public static void MXPredPartialForward(PredictorHandle handle, int step, out int step_left)
        {
            if (handle == null)
                throw new ArgumentNullException(nameof(handle));

            handle.ThrowIfDisposed();

            var ret = NativeMethods.MXPredPartialForward(handle.NativePtr, step, out step_left);
            if (ret != NativeMethods.OK)
                throw CreateMXNetException($"Failed to partial forward {nameof(handle)}");
        }

        public static void MXPredSetInput(PredictorHandle handle, string key, float[] data, uint size)
        {
            if (handle == null)
                throw new ArgumentNullException(nameof(handle));

            handle.ThrowIfDisposed();

            var ret = NativeMethods.MXPredSetInput(handle.NativePtr, key, data, size);
            if (ret != NativeMethods.OK)
                throw CreateMXNetException($"Failed to input {nameof(handle)}");
        }

        #region Helpers

        private static MXNetException CreateMXNetException(string message)
        {
            var error = NativeMethods.MXGetLastError();
            var innerMessage = Marshal.PtrToStringAnsi(error);
            var predictException = new PredictException(innerMessage);
            return new MXNetException(message, predictException);
        }

        #endregion

        #endregion

    }

}
