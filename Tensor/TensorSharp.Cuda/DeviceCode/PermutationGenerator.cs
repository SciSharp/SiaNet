// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="PermutationGenerator.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.DeviceCode
{
    /// <summary>
    /// Class PermutationGenerator.
    /// </summary>
    public class PermutationGenerator
    {
        /// <summary>
        /// The sb
        /// </summary>
        public readonly StringBuilder sb = new StringBuilder();

        /// <summary>
        /// Initializes a new instance of the <see cref="PermutationGenerator"/> class.
        /// </summary>
        public PermutationGenerator()
        {
        }

        /// <summary>
        /// Returns a <see cref="System.String" /> that represents this instance.
        /// </summary>
        /// <returns>A <see cref="System.String" /> that represents this instance.</returns>
        public override string ToString()
        {
            return sb.ToString();
        }

        /// <summary>
        /// Adds the apply t.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="operatorCode">The operator code.</param>
        public void AddApplyT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("APPLY_T({0}, {1}, {2}, {3})\n", indexType, dimsA, kernelName, operatorCode);
            }
        }

        /// <summary>
        /// Adds the apply tt.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="operatorCode">The operator code.</param>
        public void AddApplyTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("APPLY_TT({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, dimsB, kernelName, operatorCode);
            }
        }

        /// <summary>
        /// Adds the apply TTT.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="operatorCode">The operator code.</param>
        public void AddApplyTTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(3))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();
                sb.AppendFormat("APPLY_TTT({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, dimsC, kernelName, operatorCode);
            }
        }

        /// <summary>
        /// Adds the apply ts.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="operatorCode">The operator code.</param>
        public void AddApplyTS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("APPLY_TS({0}, {1}, {2}, {3})\n", indexType, dimsA, kernelName, operatorCode);
            }
        }

        /// <summary>
        /// Adds the apply TSS.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="operatorCode">The operator code.</param>
        public void AddApplyTSS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("APPLY_TSS({0}, {1}, {2}, {3})\n", indexType, dimsA, kernelName, operatorCode);
            }
        }

        /// <summary>
        /// Adds the apply TTS.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="operatorCode">The operator code.</param>
        public void AddApplyTTS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("APPLY_TTS({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, dimsB, kernelName, operatorCode);
            }
        }

        /// <summary>
        /// Adds the apply TTSS.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="operatorCode">The operator code.</param>
        public void AddApplyTTSS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("APPLY_TTSS({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, dimsB, kernelName, operatorCode);
            }
        }

        /// <summary>
        /// Adds the apply TTTS.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="operatorCode">The operator code.</param>
        public void AddApplyTTTS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(3))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();
                sb.AppendFormat("APPLY_TTTS({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, dimsC, kernelName, operatorCode);
            }
        }


        /// <summary>
        /// Adds the reduce.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="modifyOpCode">The modify op code.</param>
        /// <param name="reduceOpCode">The reduce op code.</param>
        public void AddReduce(string kernelBaseName, string modifyOpCode, string reduceOpCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("REDUCE_KERNELS({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, kernelName, modifyOpCode, reduceOpCode);
            }
        }

        /// <summary>
        /// Adds the reduce norm.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        public void AddReduceNorm(string kernelBaseName)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("REDUCE_NORM_KERNELS({0}, {1}, {2}, {3})\n", indexType, dimsA, dimsB, kernelName);
            }
        }

        /// <summary>
        /// Adds the reduce all.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        /// <param name="modifyOpCode">The modify op code.</param>
        /// <param name="reduceOpCode">The reduce op code.</param>
        public void AddReduceAll(string kernelBaseName, string modifyOpCode, string reduceOpCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("REDUCE_ALL_KERNELS({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, kernelName, modifyOpCode, reduceOpCode);
            }
        }

        /// <summary>
        /// Adds the reduce all norm.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        public void AddReduceAllNorm(string kernelBaseName)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("REDUCE_ALL_NORM_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName);
            }
        }

        /// <summary>
        /// Adds the reduce all sub square.
        /// </summary>
        /// <param name="kernelBaseName">Name of the kernel base.</param>
        public void AddReduceAllSubSquare(string kernelBaseName)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("REDUCE_ALL_SUB_SQUARE_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName);
            }
        }


        // TODO make member of ApplySpecialization
        /// <summary>
        /// Gets the name of the mangled.
        /// </summary>
        /// <param name="baseName">Name of the base.</param>
        /// <param name="spec">The spec.</param>
        /// <returns>System.String.</returns>
        public static string GetMangledName(string baseName, ApplySpecialization spec)
        {
            var sb = new StringBuilder();

            sb.Append(baseName);
            sb.Append(spec.Use32BitIndices ? "__int32" : "__int64");
            foreach (var dimSize in spec.TensorDims)
            {
                sb.Append("_").Append(dimSize.ToString().Replace('-', 'M'));
            }
            return sb.ToString();
        }
    }
}
