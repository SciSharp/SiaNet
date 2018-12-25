// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CpuFillCopyOps.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Class CpuFillCopyOps.
    /// </summary>
    [OpsClass]
    public class CpuFillCopyOps
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CpuFillCopyOps"/> class.
        /// </summary>
        public CpuFillCopyOps()
        {
        }


        /// <summary>
        /// The fill function
        /// </summary>
        private MethodInfo fill_func = NativeWrapper.GetMethod("TS_Fill");
        /// <summary>
        /// Fills the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        [RegisterOpStorageType("fill", typeof(CpuStorage))]
        public void Fill(Tensor result, float value)
        {
            NativeWrapper.InvokeTypeMatch(fill_func, result, value);
        }


        /// <summary>
        /// The copy function
        /// </summary>
        private MethodInfo copy_func = NativeWrapper.GetMethod("TS_Copy");
        /// <summary>
        /// Copies the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <exception cref="InvalidOperationException">Tensors must have equal numbers of elements</exception>
        [RegisterOpStorageType("copy", typeof(CpuStorage))]
        public void Copy(Tensor result, Tensor src)
        {
            if (result.ElementCount() != src.ElementCount())
                throw new InvalidOperationException("Tensors must have equal numbers of elements");
            NativeWrapper.Invoke(copy_func, result, src);
        }
    }
}
