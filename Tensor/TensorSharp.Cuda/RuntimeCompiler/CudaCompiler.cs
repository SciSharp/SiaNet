// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaCompiler.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    /// <summary>
    /// Class CudaCompiler.
    /// </summary>
    public class CudaCompiler
    {
        /// <summary>
        /// The includes
        /// </summary>
        private readonly Dictionary<string, string> includes = new Dictionary<string, string>();
        /// <summary>
        /// The disk cache
        /// </summary>
        private readonly KernelDiskCache diskCache;

        /// <summary>
        /// Initializes a new instance of the <see cref="CudaCompiler"/> class.
        /// </summary>
        /// <param name="diskCache">The disk cache.</param>
        public CudaCompiler(KernelDiskCache diskCache)
        {
            this.diskCache = diskCache;
            RegisterAttributeHeaders(Assembly.GetExecutingAssembly());
        }

        /// <summary>
        /// Compiles to PTX.
        /// </summary>
        /// <param name="code">The code.</param>
        /// <param name="prependIncludes">The prepend includes.</param>
        /// <returns>System.Byte[].</returns>
        public byte[] CompileToPtx(string code, params string[] prependIncludes)
        {
            // We manually prepend include files here, so that the header content forms part of the hash of the source
            // code. This means that changes to headers will correctly trigger a recompile.
            var finalCode = new StringBuilder();
            foreach (var includeName in prependIncludes)
            {
                finalCode.Append(includes[includeName]).Append('\n');
            }
            finalCode.Append(code);
            var finalCodeString = finalCode.ToString();

            return diskCache.Get(finalCodeString, DoCompile);
        }

        /// <summary>
        /// Does the compile.
        /// </summary>
        /// <param name="fullSource">The full source.</param>
        /// <returns>System.Byte[].</returns>
        /// <exception cref="ApplicationException">Error compiling CUDA code: " + rtc.GetLogAsString()</exception>
        private byte[] DoCompile(string fullSource)
        {
            var rtc = new ManagedCuda.NVRTC.CudaRuntimeCompiler(fullSource, null);

            try
            {
                rtc.Compile(new string[0]);
            }
            catch
            {
                throw new ApplicationException("Error compiling CUDA code: " + rtc.GetLogAsString());
            }

            return rtc.GetPTX();
        }

        /// <summary>
        /// Registers the header.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="content">The content.</param>
        public void RegisterHeader(string name, string content)
        {
            this.includes.Add(name, content);
        }


        /// <summary>
        /// Registers the attribute headers.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        private void RegisterAttributeHeaders(Assembly assembly)
        {
            foreach (var applyType in assembly.TypesWithAttribute<CudaIncludeAttribute>(false))
            {
                foreach(var attribute in applyType.Item2)
                {
                    var info = HeaderInfoFromAttribute(applyType.Item1, attribute);
                    RegisterHeader(info.Item1, info.Item2);
                }
            }
        }

        /// <summary>
        /// Headers the information from attribute.
        /// </summary>
        /// <param name="containingType">Type of the containing.</param>
        /// <param name="attribute">The attribute.</param>
        /// <returns>Tuple&lt;System.String, System.String&gt;.</returns>
        private Tuple<string, string> HeaderInfoFromAttribute(Type containingType, CudaIncludeAttribute attribute)
        {
            var field = containingType.GetField(attribute.FieldName, BindingFlags.Public | BindingFlags.Static);
            var content = (string)field.GetValue(null);
            return Tuple.Create(attribute.IncludeName, content);
        }
    }
}
