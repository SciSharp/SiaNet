// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="DeviceKernelTemplate.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    /// <summary>
    /// Class DeviceKernelTemplate.
    /// </summary>
    public class DeviceKernelTemplate
    {
        /// <summary>
        /// The template code
        /// </summary>
        private readonly string templateCode;
        /// <summary>
        /// The required headers
        /// </summary>
        private readonly List<string> requiredHeaders;
        /// <summary>
        /// The required configuration arguments
        /// </summary>
        private readonly HashSet<string> requiredConfigArgs = new HashSet<string>();
        /// <summary>
        /// The PTX cache
        /// </summary>
        private readonly Dictionary<KernelConfig, byte[]> ptxCache = new Dictionary<KernelConfig, byte[]>();


        /// <summary>
        /// Initializes a new instance of the <see cref="DeviceKernelTemplate"/> class.
        /// </summary>
        /// <param name="templateCode">The template code.</param>
        /// <param name="requiredHeaders">The required headers.</param>
        public DeviceKernelTemplate(string templateCode, params string[] requiredHeaders)
        {
            this.templateCode = templateCode;
            this.requiredHeaders = new List<string>(requiredHeaders);
        }

        /// <summary>
        /// Adds the configuration arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public void AddConfigArgs(params string[] args)
        {
            foreach(var item in args)
            {
                requiredConfigArgs.Add(item);
            }
        }

        /// <summary>
        /// Adds the headers.
        /// </summary>
        /// <param name="headers">The headers.</param>
        public void AddHeaders(params string[] headers)
        {
            requiredHeaders.AddRange(headers);
        }

        /// <summary>
        /// PTXs for configuration.
        /// </summary>
        /// <param name="compiler">The compiler.</param>
        /// <param name="config">The configuration.</param>
        /// <returns>System.Byte[].</returns>
        /// <exception cref="InvalidOperationException">
        /// All config arguments must be provided. Required: " + allRequired
        /// or
        /// Config provides some unnecessary arguments. Required: " + allRequired
        /// </exception>
        public byte[] PtxForConfig(CudaCompiler compiler, KernelConfig config)
        {
            byte[] cachedResult;
            if (ptxCache.TryGetValue(config, out cachedResult))
            {
                return cachedResult;
            }

            if(!requiredConfigArgs.All(config.ContainsKey))
            {
                var allRequired = string.Join(", ", requiredConfigArgs);
                throw new InvalidOperationException("All config arguments must be provided. Required: " + allRequired);
            }

            // Checking this ensures that there is only one config argument that can evaluate to the same code,
            // which ensures that the ptx cacheing does not generate unnecessary combinations. Also, a mismatch
            // occurring here probably indicates a bug somewhere else.
            if(!config.Keys.All(requiredConfigArgs.Contains))
            {
                var allRequired = string.Join(", ", requiredConfigArgs);
                throw new InvalidOperationException("Config provides some unnecessary arguments. Required: " + allRequired);
            }

            //return new DeviceKernelCode(config.ApplyToTemplate(templateCode), requiredHeaders.ToArray());
            var finalCode = config.ApplyToTemplate(templateCode);

            var result = compiler.CompileToPtx(finalCode, requiredHeaders.ToArray());
            ptxCache.Add(config, result);
            return result;
        }
    }
}
