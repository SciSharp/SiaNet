// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="TensorSerialization.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp
{
    /// <summary>
    /// Class TensorSerialization.
    /// </summary>
    public static class TensorSerialization
    {
        /// <summary>
        /// Serializes the specified tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="stream">The stream.</param>
        public static void Serialize(Tensor tensor, Stream stream)
        {
            using (var src = Ops.AsContiguous(tensor))
            {
                // Note: don't dispose writer - it does not own the stream's lifetime
                var writer = new System.IO.BinaryWriter(stream);

                // Can infer strides - src is contiguous
                writer.Write(tensor.DimensionCount); // int32
                writer.Write((int)tensor.ElementType);
                for (int i = 0; i < tensor.DimensionCount; ++i)
                {
                    writer.Write(tensor.Sizes[i]);
                }

                var byteCount = src.ElementType.Size() * tensor.ElementCount();
                writer.Write(byteCount);
                WriteBytes(writer, src.Storage, src.StorageOffset, byteCount);

                writer.Flush();
            }
        }

        /// <summary>
        /// Deserializes the specified allocator.
        /// </summary>
        /// <param name="allocator">The allocator.</param>
        /// <param name="stream">The stream.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Deserialize(IAllocator allocator, Stream stream)
        {
            // Note: don't dispose reader - it does not own the stream's lifetime
            var reader = new BinaryReader(stream);

            var dimCount = reader.ReadInt32();
            var elementType = (DType)reader.ReadInt32();
            var sizes = new long[dimCount];
            for (int i = 0; i < dimCount; ++i)
            {
                sizes[i] = reader.ReadInt64();
            }

            var byteCount = reader.ReadInt64();
            var result = new Tensor(allocator, elementType, sizes);

            ReadBytes(reader, result.Storage, result.StorageOffset, byteCount);

            return result;
        }

        /// <summary>
        /// Writes the bytes.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="storage">The storage.</param>
        /// <param name="startIndex">The start index.</param>
        /// <param name="byteCount">The byte count.</param>
        private static void WriteBytes(BinaryWriter writer, Storage storage, long startIndex, long byteCount)
        {
            var buffer = new byte[4096];
            var bufferHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                long curStart = startIndex;
                long afterLastByte = startIndex + byteCount;
                while (curStart < afterLastByte)
                {
                    var length = (int)Math.Min(buffer.Length, afterLastByte - curStart);
                    storage.CopyFromStorage(bufferHandle.AddrOfPinnedObject(), curStart, length);
                    writer.Write(buffer, 0, length);
                    curStart += length;
                }
            }
            finally
            {
                bufferHandle.Free();
            }
        }

        /// <summary>
        /// Reads the bytes.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <param name="storage">The storage.</param>
        /// <param name="startIndex">The start index.</param>
        /// <param name="byteCount">The byte count.</param>
        private static void ReadBytes(BinaryReader reader, Storage storage, long startIndex, long byteCount)
        {
            var buffer = new byte[4096];
            var bufferHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                long curStart = startIndex;
                long afterLastByte = startIndex + byteCount;
                while (curStart < afterLastByte)
                {
                    var length = (int)Math.Min(buffer.Length, afterLastByte - curStart);
                    reader.Read(buffer, 0, length);
                    storage.CopyToStorage(curStart, bufferHandle.AddrOfPinnedObject(), length);
                    curStart += length;
                }
            }
            finally
            {
                bufferHandle.Free();
            }
        }
    }
}
