// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaCode.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA.DeviceCode
{
    /// <summary>
    /// Class CudaCode.
    /// Implements the <see cref="TensorSharp.CUDA.IPrecompilable" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.IPrecompilable" />
    public abstract class CudaCode : IPrecompilable
    {
        /// <summary>
        /// The code
        /// </summary>
        private readonly string code;
        /// <summary>
        /// The required headers
        /// </summary>
        private readonly string[] requiredHeaders;
        /// <summary>
        /// The PTX
        /// </summary>
        private byte[] ptx = null;

        /// <summary>
        /// Initializes a new instance of the <see cref="CudaCode"/> class.
        /// </summary>
        /// <param name="code">The code.</param>
        /// <param name="requiredHeaders">The required headers.</param>
        protected CudaCode(string code, params string[] requiredHeaders)
        {
            this.code = code;
            this.requiredHeaders = requiredHeaders;
        }

        /// <summary>
        /// Gets the PTX.
        /// </summary>
        /// <param name="compiler">The compiler.</param>
        /// <returns>System.Byte[].</returns>
        public byte[] GetPtx(CudaCompiler compiler)
        {
            if (ptx == null)
            {
                Precompile(compiler);
            }
            return ptx;
        }

        /// <summary>
        /// Precompiles the specified compiler.
        /// </summary>
        /// <param name="compiler">The compiler.</param>
        public void Precompile(CudaCompiler compiler)
        {
            ptx = compiler.CompileToPtx(code, requiredHeaders);
        }

        public static string ReadFile(string name)
        {
            string filePath = AppDomain.CurrentDomain.BaseDirectory + "/Cuda/DeviceCode/CU/" + name;
            return File.ReadAllText(filePath, Encoding.UTF8);
        }
    }

}
