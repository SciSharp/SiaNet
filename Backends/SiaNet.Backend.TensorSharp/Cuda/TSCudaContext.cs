// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="TSCudaContext.cs" company="SiaNet.Backend.TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using SiaNet.Backend.TensorSharp.CUDA.ContextState;
using SiaNet.Backend.TensorSharp.CUDA.Util;

namespace SiaNet.Backend.TensorSharp.CUDA
{
    /// <summary>
    /// Struct ScratchSpace
    /// </summary>
    public struct ScratchSpace
    {
        /// <summary>
        /// The size
        /// </summary>
        public int size;
        /// <summary>
        /// The buffer
        /// </summary>
        public CUdeviceptr buffer;
    }

    /// <summary>
    /// Class TSCudaContext.
    /// Implements the <see cref="System.IDisposable" />
    /// </summary>
    /// <seealso cref="System.IDisposable" />
    public class TSCudaContext : IDisposable
    {
        /// <summary>
        /// The maximum dims
        /// </summary>
        public const int MaxDims = 25;
        /// <summary>
        /// The cache dir
        /// </summary>
        private const string CacheDir = @"cuda_cache\general";


        /// <summary>
        /// The device count
        /// </summary>
        private readonly int deviceCount;
        /// <summary>
        /// The devices
        /// </summary>
        public DeviceState[] devices;
        /// <summary>
        /// The P2P access
        /// </summary>
        private readonly bool[,] p2pAccess;

        /// <summary>
        /// The disk cache
        /// </summary>
        private readonly RuntimeCompiler.KernelDiskCache diskCache;

        /// <summary>
        /// The compiler
        /// </summary>
        private readonly RuntimeCompiler.CudaCompiler compiler;
        /// <summary>
        /// The kernel cache
        /// </summary>
        private readonly CudaKernelCache kernelCache = new CudaKernelCache();


        /// <summary>
        /// Initializes a new instance of the <see cref="TSCudaContext"/> class.
        /// </summary>
        public TSCudaContext()
        {
            try
            {
                this.deviceCount = CudaContext.GetDeviceCount();
            }
            catch
            {
                // CudaContext.GetDeviceCount() throws if CUDA drivers are not installed
                this.deviceCount = 0;
            }

            this.devices = Enumerable.Repeat(0, deviceCount)
                .Select(x => new DeviceState(x))
                .ToArray();

            if (deviceCount > 0)
            {
                p2pAccess = EnablePeerAccess(devices.Select(x => x.CudaContext).ToArray(), devices[0].CudaContext);
            }
            else
            {
                p2pAccess = new bool[0, 0];
            }

            this.diskCache = new RuntimeCompiler.KernelDiskCache(Path.Combine(Environment.CurrentDirectory, CacheDir));
            this.compiler = new RuntimeCompiler.CudaCompiler(diskCache);

            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        /// <summary>
        /// Gets the compiler.
        /// </summary>
        /// <value>The compiler.</value>
        public RuntimeCompiler.CudaCompiler Compiler { get { return compiler; } }
        /// <summary>
        /// Gets the kernel cache.
        /// </summary>
        /// <value>The kernel cache.</value>
        public CudaKernelCache KernelCache { get { return kernelCache; } }



        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            kernelCache.Dispose();

            foreach (var device in devices)
            {
                device.Dispose();
            }
        }

        /// <summary>
        /// Synchronizes the specified device identifier.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        public void Synchronize(int deviceId)
        {
            devices[deviceId].CudaContext.Synchronize();
        }

        /// <summary>
        /// Synchronizes all.
        /// </summary>
        public void SynchronizeAll()
        {
            foreach (var device in devices)
            {
                device.CudaContext.Synchronize();
            }
        }

        /// <summary>
        /// Cudas the context for device.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <returns>CudaContext.</returns>
        public CudaContext CudaContextForDevice(int deviceId)
        {
            return devices[deviceId].CudaContext;
        }

        /// <summary>
        /// Allocators for device.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <returns>IDeviceAllocator.</returns>
        public IDeviceAllocator AllocatorForDevice(int deviceId)
        {
            return devices[deviceId].MemoryAllocator;
        }

        /// <summary>
        /// Cudas the context for tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>CudaContext.</returns>
        public CudaContext CudaContextForTensor(NDArray tensor)
        {
            return CudaContextForDevice(CudaHelpers.GetDeviceId(tensor));
        }

        /// <summary>
        /// Scratches the space for device.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <returns>ScratchSpace.</returns>
        public ScratchSpace ScratchSpaceForDevice(int deviceId)
        {
            return devices[deviceId].ScratchSpace;
        }

        /// <summary>
        /// Blases for device.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <returns>PooledObject&lt;CudaBlas&gt;.</returns>
        public PooledObject<CudaBlas> BlasForDevice(int deviceId)
        {
            return devices[deviceId].BlasHandles.Get();
        }

        /// <summary>
        /// Blases for tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>PooledObject&lt;CudaBlas&gt;.</returns>
        public PooledObject<CudaBlas> BlasForTensor(NDArray tensor)
        {
            return BlasForDevice(CudaHelpers.GetDeviceId(tensor));
        }

        /// <summary>
        /// DNNs for tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>PooledObject&lt;ManagedCuda.CudaDNN.CudaDNNContext&gt;.</returns>
        public PooledObject<ManagedCuda.CudaDNN.CudaDNNContext> DNNForTensor(NDArray tensor)
        {
            var deviceId = CudaHelpers.GetDeviceId(tensor);
            return devices[deviceId].DnnHandles.Get();
        }

        /// <summary>
        /// DNNs for device.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <returns>PooledObject&lt;ManagedCuda.CudaDNN.CudaDNNContext&gt;.</returns>
        public PooledObject<ManagedCuda.CudaDNN.CudaDNNContext> DNNForDevice(int deviceId)
        {
            return devices[deviceId].DnnHandles.Get();
        }

        /// <summary>
        /// Determines whether this instance [can access peer] the specified source device.
        /// </summary>
        /// <param name="srcDevice">The source device.</param>
        /// <param name="peerDevice">The peer device.</param>
        /// <returns><c>true</c> if this instance [can access peer] the specified source device; otherwise, <c>false</c>.</returns>
        public bool CanAccessPeer(int srcDevice, int peerDevice)
        {
            return p2pAccess[srcDevice, peerDevice];
        }

        /// <summary>
        /// Devices the information for context.
        /// </summary>
        /// <param name="cudaContext">The cuda context.</param>
        /// <returns>CudaDeviceProperties.</returns>
        public CudaDeviceProperties DeviceInfoForContext(CudaContext cudaContext)
        {
            return devices[cudaContext.DeviceId].DeviceInfo;
        }



        // Returns a matrix of [i, j] values where [i, j] is true iff device i can access device j
        /// <summary>
        /// Enables the peer access.
        /// </summary>
        /// <param name="cudaContexts">The cuda contexts.</param>
        /// <param name="restoreCurrent">The restore current.</param>
        /// <returns>System.Boolean[].</returns>
        private static bool[,] EnablePeerAccess(CudaContext[] cudaContexts, CudaContext restoreCurrent)
        {
            var result = new bool[cudaContexts.Length, cudaContexts.Length];

            for (int i = 0; i < cudaContexts.Length; ++i)
            {
                for (int j = 0; j < cudaContexts.Length; ++j)
                {
                    if (i == j)
                    {
                        result[i, j] = true;
                    }
                    else
                    {
                        result[i, j] = EnablePeers(cudaContexts[i], cudaContexts[j]);
                    }
                }
            }

            restoreCurrent.SetCurrent();
            return result;
        }

        /// <summary>
        /// Enables the peers.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="target">The target.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        private static bool EnablePeers(CudaContext src, CudaContext target)
        {
            if (!src.DeviceCanAccessPeer(target))
                return false;

            src.SetCurrent();

            try
            {
                CudaContext.EnablePeerAccess(target);
                return true;
            }
            catch
            {
                return false;
            }
        }


        /// <summary>
        /// Precompiles this instance.
        /// </summary>
        public void Precompile()
        {
            Precompile(Console.Write);
        }

        /// <summary>
        /// Precompiles the specified precompile progress writer.
        /// </summary>
        /// <param name="precompileProgressWriter">The precompile progress writer.</param>
        public void Precompile(Action<string> precompileProgressWriter)
        {
            var assembly = Assembly.GetExecutingAssembly();
            foreach (var applyType in assembly.TypesWithAttribute<PrecompileAttribute>(true).Where(x => !x.Item1.IsAbstract))
            {
                precompileProgressWriter("Precompiling " + applyType.Item1.Name + "\n");

                var instance = (IPrecompilable)Activator.CreateInstance(applyType.Item1);
                instance.Precompile(Compiler);
            }
        }

        /// <summary>
        /// Cleans the unused PTX.
        /// </summary>
        public void CleanUnusedPTX()
        {
            diskCache.CleanUnused();
        }
    }
}
