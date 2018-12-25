// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="KernelDiskCache.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    /// <summary>
    /// Class KernelDiskCache.
    /// </summary>
    public class KernelDiskCache
    {
        /// <summary>
        /// The cache dir
        /// </summary>
        private readonly string cacheDir;
        /// <summary>
        /// The memory cached kernels
        /// </summary>
        private readonly Dictionary<string, byte[]> memoryCachedKernels = new Dictionary<string, byte[]>();


        /// <summary>
        /// Initializes a new instance of the <see cref="KernelDiskCache"/> class.
        /// </summary>
        /// <param name="cacheDir">The cache dir.</param>
        public KernelDiskCache(string cacheDir)
        {
            this.cacheDir = cacheDir;
            if (!System.IO.Directory.Exists(cacheDir))
            {
                System.IO.Directory.CreateDirectory(cacheDir);
            }
        }

        /// <summary>
        /// Deletes all kernels from disk if they are not currently loaded into memory. Calling this after
        /// calling TSCudaContext.Precompile() will delete any cached .ptx files that are no longer needed
        /// </summary>
        public void CleanUnused()
        {
            foreach (var file in Directory.GetFiles(cacheDir))
            {
                var key = KeyFromFilePath(file);
                if (!memoryCachedKernels.ContainsKey(key))
                {
                    File.Delete(file);
                }
            }
        }

        /// <summary>
        /// Gets the specified full source code.
        /// </summary>
        /// <param name="fullSourceCode">The full source code.</param>
        /// <param name="compile">The compile.</param>
        /// <returns>System.Byte[].</returns>
        public byte[] Get(string fullSourceCode, Func<string, byte[]> compile)
        {
            var key = KeyFromSource(fullSourceCode);
            byte[] ptx;
            if (memoryCachedKernels.TryGetValue(key, out ptx))
            {
                return ptx;
            }
            else if (TryGetFromFile(key, out ptx))
            {
                memoryCachedKernels.Add(key, ptx);
                return ptx;
            }
            else
            {
                ptx = compile(fullSourceCode);
                memoryCachedKernels.Add(key, ptx);
                WriteToFile(key, ptx);
                return ptx;
            }
        }


        /// <summary>
        /// Writes to file.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <param name="ptx">The PTX.</param>
        private void WriteToFile(string key, byte[] ptx)
        {
            var filePath = FilePathFromKey(key);
            System.IO.File.WriteAllBytes(filePath, ptx);
        }

        /// <summary>
        /// Tries the get from file.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <param name="ptx">The PTX.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        private bool TryGetFromFile(string key, out byte[] ptx)
        {
            var filePath = FilePathFromKey(key);
            if (!System.IO.File.Exists(filePath))
            {
                ptx = null;
                return false;
            }

            ptx = System.IO.File.ReadAllBytes(filePath);
            return true;
        }

        /// <summary>
        /// Files the path from key.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <returns>System.String.</returns>
        private string FilePathFromKey(string key)
        {
            return System.IO.Path.Combine(cacheDir, key + ".ptx");
        }

        /// <summary>
        /// Keys from file path.
        /// </summary>
        /// <param name="filepath">The filepath.</param>
        /// <returns>System.String.</returns>
        private string KeyFromFilePath(string filepath)
        {
            return Path.GetFileNameWithoutExtension(filepath);
        }

        /// <summary>
        /// Keys from source.
        /// </summary>
        /// <param name="fullSource">The full source.</param>
        /// <returns>System.String.</returns>
        private static string KeyFromSource(string fullSource)
        {
            var fullKey = fullSource.Length.ToString() + fullSource;

            using (var sha1 = new SHA1Managed())
            {
                return BitConverter.ToString(sha1.ComputeHash(Encoding.UTF8.GetBytes(fullKey)))
                    .Replace("-", "");
            }
        }
    }
}
