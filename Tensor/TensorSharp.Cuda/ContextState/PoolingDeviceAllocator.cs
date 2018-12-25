// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="PoolingDeviceAllocator.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ManagedCuda.BasicTypes;
using ManagedCuda;

namespace TensorSharp.CUDA.ContextState
{
    /// <summary>
    /// Class PoolingDeviceAllocator.
    /// Implements the <see cref="TensorSharp.CUDA.ContextState.IDeviceAllocator" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.ContextState.IDeviceAllocator" />
    public class PoolingDeviceAllocator : IDeviceAllocator
    {
        /// <summary>
        /// The memory alignment
        /// </summary>
        private const long MemoryAlignment = 256;

        /// <summary>
        /// The context
        /// </summary>
        private readonly CudaContext context;
        /// <summary>
        /// The pools
        /// </summary>
        private Dictionary<long, Queue<IDeviceMemory>> pools = new Dictionary<long, Queue<IDeviceMemory>>();


        /// <summary>
        /// Initializes a new instance of the <see cref="PoolingDeviceAllocator"/> class.
        /// </summary>
        /// <param name="context">The context.</param>
        public PoolingDeviceAllocator(CudaContext context)
        {
            this.context = context;
        }

        /// <summary>
        /// Allocates the specified byte count.
        /// </summary>
        /// <param name="byteCount">The byte count.</param>
        /// <returns>IDeviceMemory.</returns>
        public IDeviceMemory Allocate(long byteCount)
        {
            var size = PadToAlignment(byteCount, MemoryAlignment);

            Queue<IDeviceMemory> sizedPool;
            if (pools.TryGetValue(size, out sizedPool))
            {
                if (sizedPool.Count > 0)
                {
                    var result = sizedPool.Dequeue();

                    // HACK  bizarrely, Queue.Dequeue appears to sometimes return null, even when there are many elements in the queue,
                    // and when the queue is only ever accessed from one thread.
                    if(result != null)
                        return result;
                }
            }
            else
            {
                sizedPool = new Queue<IDeviceMemory>();
                pools.Add(size, sizedPool);
            }

            // If control flow gets to this point, sizedPool exists in the dictionary and is empty.

            var buffer = context.AllocateMemory(size);
            BasicDeviceMemory devMemory = null;
            devMemory = new BasicDeviceMemory(buffer, () =>
            {
                sizedPool.Enqueue(devMemory);
            });

            return devMemory;
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            foreach (var kvp in pools)
            {
                foreach (var item in kvp.Value)
                {
                    item.Free();
                }
            }

            pools.Clear();
        }

        /// <summary>
        /// Pads to alignment.
        /// </summary>
        /// <param name="size">The size.</param>
        /// <param name="alignment">The alignment.</param>
        /// <returns>System.Int64.</returns>
        private static long PadToAlignment(long size, long alignment)
        {
            return ((size + alignment - 1) / alignment) * alignment;
        }
    }
}
