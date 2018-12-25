// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ElementwiseTOp.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Core;
using TensorSharp.CUDA.DeviceCode;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA.KernelOps
{
    /// <summary>
    /// Class ApplyOpInvoke.
    /// </summary>
    public static class ApplyOpInvoke
    {
        /// <summary>
        /// Invokes the specified context.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="cudaContext">The cuda context.</param>
        /// <param name="ptx">The PTX.</param>
        /// <param name="baseName">Name of the base.</param>
        /// <param name="args">The arguments.</param>
        public static void Invoke(TSCudaContext context, CudaContext cudaContext, byte[] ptx, string baseName, params object[] args)
        {
            ThrowIfAnyTensorInvalid(args);

            var deviceInfo = context.DeviceInfoForContext(cudaContext);

            var allTensors = args.OfType<Tensor>();
            var firstTensor = allTensors.First();
            var elementCount = firstTensor.ElementCount();
            var spec = new ApplySpecialization(allTensors.ToArray());

            ConvertTensorArgs.Convert(cudaContext, spec.Use32BitIndices, args);
            
            var block = ApplyUtils.GetApplyBlock();
            var grid = ApplyUtils.GetApplyGrid(deviceInfo, elementCount);

            var fullKernelName = PermutationGenerator.GetMangledName(baseName, spec);
            var kernel = context.KernelCache.Get(cudaContext, ptx, fullKernelName);

            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.RunAsync(CUstream.NullStream, args);
            
        }


        /// <summary>
        /// Throws if any tensor invalid.
        /// </summary>
        /// <param name="args">The arguments.</param>
        /// <exception cref="InvalidOperationException">Tensors with dimension count > " + TSCudaContext.MaxDims + " are not supported</exception>
        private static void ThrowIfAnyTensorInvalid(object[] args)
        {
            foreach (var tensor in args.OfType<Tensor>())
            {
                if (tensor.DimensionCount > TSCudaContext.MaxDims)
                    throw new InvalidOperationException("Tensors with dimension count > " + TSCudaContext.MaxDims + " are not supported");
            }
        }


        /// <summary>
        /// Applies the precompile.
        /// </summary>
        /// <param name="compiler">The compiler.</param>
        /// <param name="template">The template.</param>
        /// <param name="tensorCount">The tensor count.</param>
        public static void ApplyPrecompile(CudaCompiler compiler, DeviceKernelTemplate template, int tensorCount)
        {
            foreach(var spec in ApplySpecialization.AllSpecializations(tensorCount))
            {
                template.PtxForConfig(compiler, spec.GetConfig());
            }
        }
    }

    /// <summary>
    /// Class ElementwiseTTOp.
    /// </summary>
    public static class ElementwiseTTOp
    {
        /// <summary>
        /// Invokes the specified kernels.
        /// </summary>
        /// <param name="kernels">The kernels.</param>
        /// <param name="funcName">Name of the function.</param>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor src)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, elementCount);
            
            return writeTarget;
        }
    }

    /// <summary>
    /// Class ElementwiseTTSOp.
    /// </summary>
    public static class ElementwiseTTSOp
    {
        /// <summary>
        /// Invokes the specified kernels.
        /// </summary>
        /// <param name="kernels">The kernels.</param>
        /// <param name="funcName">Name of the function.</param>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor src, float value)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, value, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, value, elementCount);

            return writeTarget;
        }
    }

    /// <summary>
    /// Class ElementwiseTTTOp.
    /// </summary>
    public static class ElementwiseTTTOp
    {
        /// <summary>
        /// Invokes the specified kernels.
        /// </summary>
        /// <param name="kernels">The kernels.</param>
        /// <param name="funcName">Name of the function.</param>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor lhs, Tensor rhs)
        {
            var context = CudaHelpers.TSContextForTensor(lhs);
            var cudaContext = context.CudaContextForTensor(lhs);

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == lhs)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, rhs, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, lhs, rhs, elementCount);

            return writeTarget;
        }
    }

    /// <summary>
    /// Class ClampOp.
    /// </summary>
    public static class ClampOp
    {
        /// <summary>
        /// Invokes the specified kernels.
        /// </summary>
        /// <param name="kernels">The kernels.</param>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Invoke(ElementwiseKernels kernels, Tensor result, Tensor src, float min, float max)
        {
            var funcName = "clamp";
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, min, max, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, min, max, elementCount);

            return writeTarget;
        }
    }

    /// <summary>
    /// Class Atan2Op.
    /// </summary>
    public static class Atan2Op
    {
        /// <summary>
        /// Invokes the specified kernels.
        /// </summary>
        /// <param name="kernels">The kernels.</param>
        /// <param name="result">The result.</param>
        /// <param name="srcY">The source y.</param>
        /// <param name="srcX">The source x.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Invoke(ElementwiseKernels kernels, Tensor result, Tensor srcY, Tensor srcX)
        {
            var context = CudaHelpers.TSContextForTensor(srcY);
            var cudaContext = context.CudaContextForTensor(srcY);

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, srcY, false, srcY.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);
            ApplyOpInvoke.Invoke(context, cudaContext, ptx, "atan2", writeTarget, srcY, srcX, elementCount);

            return writeTarget;
        }
    }

    /// <summary>
    /// Class LerpOp.
    /// </summary>
    public static class LerpOp
    {
        /// <summary>
        /// Invokes the specified kernels.
        /// </summary>
        /// <param name="kernels">The kernels.</param>
        /// <param name="result">The result.</param>
        /// <param name="srcA">The source a.</param>
        /// <param name="srcB">The source b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Invoke(ElementwiseKernels kernels, Tensor result, Tensor srcA, Tensor srcB, float weight)
        {
            var context = CudaHelpers.TSContextForTensor(srcA);
            var cudaContext = context.CudaContextForTensor(srcA);

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, srcA, false, srcA.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);
            ApplyOpInvoke.Invoke(context, cudaContext, ptx, "lerp", writeTarget, srcA, srcB, weight, elementCount);

            return writeTarget;
        }
    }

    /// <summary>
    /// Class CopyOp.
    /// </summary>
    public static class CopyOp
    {
        /// <summary>
        /// Invokes the specified kernels.
        /// </summary>
        /// <param name="kernels">The kernels.</param>
        /// <param name="context">The context.</param>
        /// <param name="cudaContext">The cuda context.</param>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        public static void Invoke(FillCopyKernels kernels, TSCudaContext context, CudaContext cudaContext, Tensor result, Tensor src)
        {
            var ptx = kernels.GetPtx(context.Compiler);
            var elementCount = result.ElementCount();
            ApplyOpInvoke.Invoke(context, cudaContext, ptx, "copy", result, src, elementCount);
        }
    }

    /// <summary>
    /// Class FillOp.
    /// </summary>
    public static class FillOp
    {
        /// <summary>
        /// Invokes the specified kernels.
        /// </summary>
        /// <param name="kernels">The kernels.</param>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        public static void Invoke(FillCopyKernels kernels, Tensor result, float value)
        {
            var context = CudaHelpers.TSContextForTensor(result);
            var cudaContext = context.CudaContextForTensor(result);
            var ptx = kernels.GetPtx(context.Compiler);
            var elementCount = result.ElementCount();
            ApplyOpInvoke.Invoke(context, cudaContext, ptx, "fill", result, value, elementCount);
        }
    }
}
