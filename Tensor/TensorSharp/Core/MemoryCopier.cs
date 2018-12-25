// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-26-2018
// ***********************************************************************
// <copyright file="MemoryCopier.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace TensorSharp.Core
{
    using System;
    using System.Runtime.InteropServices;

    /// <summary>
    /// Class with efficient memory copy methods
    /// </summary>
    internal static class MemoryCopier
    {
        /// <summary>
        /// Native Methods 32.
        /// </summary>
        private static class NativeMethods32
        {
            /// <summary>
            /// RTLs the copy memory.
            /// </summary>
            /// <param name="destination">The destination.</param>
            /// <param name="source">The source.</param>
            /// <param name="length">The length.</param>
            [DllImport("kernel32.dll")]
            public static extern void RtlCopyMemory(IntPtr destination, IntPtr source, uint length);
        }

        /// <summary>
        /// Native Methods 64.
        /// </summary>
        private static class NativeMethods64
        {
            /// <summary>
            /// RTLs the copy memory.
            /// </summary>
            /// <param name="destination">The destination.</param>
            /// <param name="source">The source.</param>
            /// <param name="length">The length.</param>
            [DllImport("kernel32.dll")]
            public static extern void RtlCopyMemory(IntPtr destination, IntPtr source, ulong length);
        }

        /// <summary>
        /// Copies data from source to destination.
        /// </summary>
        /// <param name="destination">The destination memory pointer.</param>
        /// <param name="source">The source memory pointer.</param>
        /// <param name="length">The length of data.</param>
        public static void Copy(IntPtr destination, IntPtr source, ulong length)
        {
            var is32 = IntPtr.Size == 4;

            if (is32)
            {
                // Note: if this is run, length should always be in range of a uint
                // (it should be impossible to allocate a buffer bigger than that range
                // on a 32-bit system)
                NativeMethods32.RtlCopyMemory(destination, source, (uint)length);
            }
            else
            {
                NativeMethods64.RtlCopyMemory(destination, source, (ulong)length);
            }

        }
    }
}
