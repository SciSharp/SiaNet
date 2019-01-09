// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="DeviceKernel.cs" company="TensorSharp.CUDA91">
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
    /// Class DeviceKernel.
    /// </summary>
    public class DeviceKernel
    {
        /// <summary>
        /// The PTX
        /// </summary>
        private readonly byte[] ptx;


        /// <summary>
        /// Initializes a new instance of the <see cref="DeviceKernel"/> class.
        /// </summary>
        /// <param name="ptx">The PTX.</param>
        public DeviceKernel(byte[] ptx)
        {
            this.ptx = ptx;
        }


    }
}
