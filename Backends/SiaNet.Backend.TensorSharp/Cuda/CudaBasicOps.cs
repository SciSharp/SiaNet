// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaBasicOps.cs" company="SiaNet.Backend.TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SiaNet.Backend.TensorSharp.Core;
using SiaNet.Backend.TensorSharp.CUDA.DeviceCode;
using SiaNet.Backend.TensorSharp.CUDA.KernelOps;
using SiaNet.Backend.TensorSharp.CUDA.MatrixMul;
using SiaNet.Backend.TensorSharp.CUDA.RuntimeCompiler;

namespace SiaNet.Backend.TensorSharp.CUDA
{
    /// <summary>
    /// Class CudaBasicOps.
    /// </summary>
    [OpsClass]
    public class CudaBasicOps
    {
        /// <summary>
        /// The copy ops
        /// </summary>
        private readonly CopyOps copyOps;

        /// <summary>
        /// The elementwise kernels
        /// </summary>
        private readonly ElementwiseKernels elementwiseKernels = new ElementwiseKernels();
        /// <summary>
        /// The fill copy kernels
        /// </summary>
        private readonly FillCopyKernels fillCopyKernels = new FillCopyKernels();

        /// <summary>
        /// The cuda reduce kernels
        /// </summary>
        private readonly CudaReduceKernels cudaReduceKernels = new CudaReduceKernels();
        /// <summary>
        /// The cuda reduce all kernels
        /// </summary>
        private readonly CudaReduceAllKernels cudaReduceAllKernels = new CudaReduceAllKernels();

        /// <summary>
        /// The variable standard kernels
        /// </summary>
        private readonly VarStdKernels varStdKernels = new VarStdKernels();
        /// <summary>
        /// The reduce dim index kernels
        /// </summary>
        private readonly ReduceDimIndexKernels reduceDimIndexKernels = new ReduceDimIndexKernels();


        /// <summary>
        /// Initializes a new instance of the <see cref="CudaBasicOps"/> class.
        /// </summary>
        public CudaBasicOps()
        {
            this.copyOps = new CopyOps(fillCopyKernels);
        }


        /*
        public Tensor NewContiguous(Tensor src)
        {
            var result = new Tensor(src.Allocator, src.ElementType, (long[])src.Sizes.Clone());
            Copy(result, src);
            return result;
        }

        public Tensor AsContiguous(Tensor src)
        {
            if (src.IsContiguous())
                return src.CopyRef();
            else
                return NewContiguous(src);
        }

        public Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            return TensorConcatenation.Concat(result, dimension, inputs);
        }


        public float SumAll(Tensor src) { using (var resultTensor = SumAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float ProdAll(Tensor src) { using (var resultTensor = ProdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float MinAll(Tensor src) { using (var resultTensor = MinAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float MaxAll(Tensor src) { using (var resultTensor = MaxAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }

        public float MeanAll(Tensor src) { using (var resultTensor = MeanAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float VarAll(Tensor src) { using (var resultTensor = VarAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float StdAll(Tensor src) { using (var resultTensor = StdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float NormAll(Tensor src, float value) { using (var resultTensor = NormAll(null, src, value)) { return resultTensor.GetElementAsFloat(0); } }

        */


        /// <summary>
        /// Copies the gpu.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <exception cref="InvalidOperationException">Tensors must have equal numbers of elements</exception>
        [RegisterOpArgCount("copy")]
        public void CopyGpu(
            [OpArgStorageType(typeof(CudaStorage))] NDArray result,
            [OpArgStorageType(typeof(CudaStorage))] NDArray src)
        {
            var totalElements = result.ElementCount();
            if (totalElements != src.ElementCount())
                throw new InvalidOperationException("Tensors must have equal numbers of elements");

            if (src.DimensionCount == 0) return;
            
            copyOps.CopyGpu(result, src, totalElements);
        }

        /// <summary>
        /// Copies the cpu to gpu.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <exception cref="InvalidOperationException">Tensors must have equal numbers of elements</exception>
        [RegisterOpArgCount("copy")]
        public void CopyCpuToGpu(
            [OpArgStorageType(typeof(CudaStorage))] NDArray result,
            [OpArgStorageType(typeof(Cpu.CpuStorage))] NDArray src)
        {
            var totalElements = result.ElementCount();
            if (totalElements != src.ElementCount())
                throw new InvalidOperationException("Tensors must have equal numbers of elements");

            if (src.DimensionCount == 0) return;

            copyOps.CopyCpuToGpu(result, src, totalElements);
        }

        /// <summary>
        /// Copies the gpu to cpu.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <exception cref="InvalidOperationException">Tensors must have equal numbers of elements</exception>
        [RegisterOpArgCount("copy")]
        public void CopyGpuToCpu(
            [OpArgStorageType(typeof(Cpu.CpuStorage))] NDArray result,
            [OpArgStorageType(typeof(CudaStorage))] NDArray src)
        {
            var totalElements = result.ElementCount();
            if (totalElements != src.ElementCount())
                throw new InvalidOperationException("Tensors must have equal numbers of elements");

            if (src.DimensionCount == 0) return;

            copyOps.CopyGpuToCpu(result, src, totalElements);
        }


        /// <summary>
        /// Fills the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        [RegisterOpStorageType("fill", typeof(CudaStorage))]
        public void Fill(NDArray result, float value)
        {
            FillOp.Invoke(fillCopyKernels, result, value);
        }


        /// <summary>
        /// Dots the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="NotSupportedException"></exception>
        [RegisterOpStorageType("dot", typeof(CudaStorage))]
        public NDArray Dot(NDArray result, NDArray lhs, NDArray rhs)
        {
            var context = CudaHelpers.TSContextForTensor(lhs);
            if (lhs.DimensionCount == 1 && rhs.DimensionCount == 1)
            {
                return CudaMatrixMulDot.Dot(context, result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && (rhs.DimensionCount == 1 || rhs.PossibleVector))
            {
                return CudaMatrixMulMV.Mul_M_V(context, result, lhs, rhs.Ravel()).Reshape(lhs.Shape[0], 1);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 2)
            {
                return CudaMatrixMulMM.Mul_M_M(context, result, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException(string.Format("Multiplication of {0}D with {1}D tensor is not supported"));
            }
        }

        /// <summary>
        /// Addmms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="src">The source.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="m1">The m1.</param>
        /// <param name="m2">The m2.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="InvalidOperationException">
        /// All tensors must have the same element type
        /// or
        /// Size mismatch
        /// </exception>
        /// <exception cref="ArgumentException">
        /// result must be a CUDA tensor - result
        /// or
        /// m1 must be a CUDA tensor - m1
        /// or
        /// m2 must be a CUDA tensor - m2
        /// or
        /// src must be a matrix - src
        /// or
        /// m1 must be a matrix - m1
        /// or
        /// m2 must be a matrix - m2
        /// </exception>
        [RegisterOpStorageType("addmm", typeof(CudaStorage))]
        public NDArray Addmm(NDArray result, float beta, NDArray src, float alpha, NDArray m1, NDArray m2)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            if (result != null && !(result.Storage is CudaStorage)) throw new ArgumentException("result must be a CUDA tensor", "result");
            if (!(m1.Storage is CudaStorage)) throw new ArgumentException("m1 must be a CUDA tensor", "m1");
            if (!(m2.Storage is CudaStorage)) throw new ArgumentException("m2 must be a CUDA tensor", "m2");

            if (src.DimensionCount != 2) throw new ArgumentException("src must be a matrix", "src");
            if (m1.DimensionCount != 2) throw new ArgumentException("m1 must be a matrix", "m1");
            if (m2.DimensionCount != 2) throw new ArgumentException("m2 must be a matrix", "m2");

            if (src.Shape[0] != m1.Shape[0] || src.Shape[1] != m2.Shape[1] || m1.Shape[1] != m2.Shape[0])
                throw new InvalidOperationException("Size mismatch");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Shape);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }
            
            CudaMatrixMulMM.Gemm(context, alpha, m1, m2, beta, writeTarget);
           

            return writeTarget;
        }


        /// <summary>
        /// Abses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("abs", typeof(CudaStorage))]
        public NDArray Abs(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "abs", result, src); }
        /// <summary>
        /// Negs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("neg", typeof(CudaStorage))]
        public NDArray Neg(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "neg", result, src); }
        /// <summary>
        /// Signs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sign", typeof(CudaStorage))]
        public NDArray Sign(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sign", result, src); }

        /// <summary>
        /// SQRTs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sqrt", typeof(CudaStorage))]
        public NDArray Sqrt(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sqrt", result, src); }
        /// <summary>
        /// Exps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("exp", typeof(CudaStorage))]
        public NDArray Exp(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "exp", result, src); }
        /// <summary>
        /// Logs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("log", typeof(CudaStorage))]
        public NDArray Log(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "log", result, src); }
        /// <summary>
        /// Log1ps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("log1p", typeof(CudaStorage))]
        public NDArray Log1p(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "log1p", result, src); }
        /// <summary>
        /// Floors the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("floor", typeof(CudaStorage))]
        public NDArray Floor(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "floor", result, src); }
        /// <summary>
        /// Ceils the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("ceil", typeof(CudaStorage))]
        public NDArray Ceil(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "ceil", result, src); }
        /// <summary>
        /// Rounds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("round", typeof(CudaStorage))]
        public NDArray Round(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "round", result, src); }
        /// <summary>
        /// Truncs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("trunc", typeof(CudaStorage))]
        public NDArray Trunc(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "trunc", result, src); }
        /// <summary>
        /// Fracs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("frac", typeof(CudaStorage))]
        public NDArray Frac(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "frac", result, src); }

        /// <summary>
        /// Sins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sin", typeof(CudaStorage))]
        public NDArray Sin(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sin", result, src); }
        /// <summary>
        /// Coses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("cos", typeof(CudaStorage))]
        public NDArray Cos(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "cos", result, src); }
        /// <summary>
        /// Tans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("tan", typeof(CudaStorage))]
        public NDArray Tan(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "tan", result, src); }

        /// <summary>
        /// Asins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("asin", typeof(CudaStorage))]
        public NDArray Asin(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "asin", result, src); }
        /// <summary>
        /// Acoses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("acos", typeof(CudaStorage))]
        public NDArray Acos(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "acos", result, src); }
        /// <summary>
        /// Atans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("atan", typeof(CudaStorage))]
        public NDArray Atan(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "atan", result, src); }

        /// <summary>
        /// Sinhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sinh", typeof(CudaStorage))]
        public NDArray Sinh(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sinh", result, src); }
        /// <summary>
        /// Coshes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("cosh", typeof(CudaStorage))]
        public NDArray Cosh(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "cosh", result, src); }
        /// <summary>
        /// Tanhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("tanh", typeof(CudaStorage))]
        public NDArray Tanh(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "tanh", result, src); }

        /// <summary>
        /// Sigmoids the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sigmoid", typeof(CudaStorage))]
        public NDArray Sigmoid(NDArray result, NDArray src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sigmoid", result, src); }


        /// <summary>
        /// Atan2s the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcY">The source y.</param>
        /// <param name="srcX">The source x.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("atan2", typeof(CudaStorage))]
        public NDArray Atan2(NDArray result, NDArray srcY, NDArray srcX) { return Atan2Op.Invoke(elementwiseKernels, result, srcY, srcX); }
        /// <summary>
        /// Pows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("pow", typeof(CudaStorage))]
        public NDArray Pow(NDArray result, NDArray src, float value) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "pow", result, src, value); }
        /// <summary>
        /// Tpows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("tpow", typeof(CudaStorage))]
        public NDArray Tpow(NDArray result, float value, NDArray src) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "tpow", result, src, value); }
        /// <summary>
        /// Lerps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcA">The source a.</param>
        /// <param name="srcB">The source b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("lerp", typeof(CudaStorage))]
        public NDArray Lerp(NDArray result, NDArray srcA, NDArray srcB, float weight) { return LerpOp.Invoke(elementwiseKernels, result, srcA, srcB, weight); }
        /// <summary>
        /// Clamps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("clamp", typeof(CudaStorage))]
        public NDArray Clamp(NDArray result, NDArray src, float min, float max) { return ClampOp.Invoke(elementwiseKernels, result, src, min, max); }

        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("addv", typeof(CudaStorage))]
        public NDArray Add(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "add", result, rhs, lhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("subv", typeof(CudaStorage))]
        public NDArray Sub(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "sub", result, rhs, lhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("rsubv", typeof(CudaStorage))]
        public NDArray Sub(NDArray result, float rhs, NDArray lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "rsub", result, lhs, rhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("mulv", typeof(CudaStorage))]
        public NDArray Mul(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "mul", result, rhs, lhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("divv", typeof(CudaStorage))]
        public NDArray Div(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "div", result, rhs, lhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("rdivv", typeof(CudaStorage))]
        public NDArray Div(NDArray result, float rhs, NDArray lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "rdiv", result, lhs, rhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("modv", typeof(CudaStorage))]
        public NDArray Mod(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "mod", result, rhs, lhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("gtValue", typeof(CudaStorage))]
        public NDArray GreaterThan(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "gt", result, rhs, lhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("ltValue", typeof(CudaStorage))]
        public NDArray LessThan(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "lt", result, rhs, lhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("geValue", typeof(CudaStorage))]
        public NDArray GreaterOrEqual(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "ge", result, rhs, lhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("leValue", typeof(CudaStorage))]
        public NDArray LessOrEqual(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "le", result, rhs, lhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("eqValue", typeof(CudaStorage))]
        public NDArray EqualTo(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "eq", result, rhs, lhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("neValue", typeof(CudaStorage))]
        public NDArray NotEqual(NDArray result, NDArray rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "ne", result, rhs, lhs); }


        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("addt", typeof(CudaStorage))]
        public NDArray Add(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cadd", result, rhs, lhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("subt", typeof(CudaStorage))]
        public NDArray Sub(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "csub", result, rhs, lhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("mult", typeof(CudaStorage))]
        public NDArray Mul(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cmul", result, rhs, lhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("divt", typeof(CudaStorage))]
        public NDArray Div(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cdiv", result, rhs, lhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("modt", typeof(CudaStorage))]
        public NDArray Mod(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cmod", result, rhs, lhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("gtTensor", typeof(CudaStorage))]
        public NDArray GreaterThan(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cgt", result, rhs, lhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("ltTensor", typeof(CudaStorage))]
        public NDArray LessThan(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "clt", result, rhs, lhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("geTensor", typeof(CudaStorage))]
        public NDArray GreaterOrEqual(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cge", result, rhs, lhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("leTensor", typeof(CudaStorage))]
        public NDArray LessOrEqual(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cle", result, rhs, lhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("eqTensor", typeof(CudaStorage))]
        public NDArray EqualTo(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "ceq", result, rhs, lhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="rhs">The RHS.</param>
        /// <param name="lhs">The LHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("neTensor", typeof(CudaStorage))]
        public NDArray NotEqual(NDArray result, NDArray rhs, NDArray lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cne", result, rhs, lhs); }


        /// <summary>
        /// Sums the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sum", typeof(CudaStorage))]
        public NDArray Sum(NDArray result, NDArray src, int dimension) { return ReductionOp.Invoke(cudaReduceKernels, "sum", 0.0f, ReduceInitType.GivenValue, result, src, dimension); }
        /// <summary>
        /// Products the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("prod", typeof(CudaStorage))]
        public NDArray Prod(NDArray result, NDArray src, int dimension) { return ReductionOp.Invoke(cudaReduceKernels, "prod", 1.0f, ReduceInitType.GivenValue, result, src, dimension); }
        /// <summary>
        /// Determines the minimum of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("min", typeof(CudaStorage))]
        public NDArray Min(NDArray result, NDArray src, int dimension) { return ReductionOp.Invoke(cudaReduceKernels, "min", 0.0f, ReduceInitType.MaxValue, result, src, dimension); }
        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("max", typeof(CudaStorage))]
        public NDArray Max(NDArray result, NDArray src, int dimension) { return ReductionOp.Invoke(cudaReduceKernels, "max", 0.0f, ReduceInitType.MinValue, result, src, dimension); }

        /// <summary>
        /// Argmins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("argmin", typeof(CudaStorage))]
        public NDArray Argmin(NDArray result, NDArray src, int dimension) { return reduceDimIndexKernels.ArgMin(result, src, dimension); }

        /// <summary>
        /// Argmaxes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("argmax", typeof(CudaStorage))]
        public NDArray Argmax(NDArray result, NDArray src, int dimension) { return reduceDimIndexKernels.ArgMax(result, src, dimension); }


        /// <summary>
        /// Means the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("mean", typeof(CudaStorage))]
        public NDArray Mean(NDArray result, NDArray src, int dimension)
        {
            var requiredOutputSize = (long[])src.Shape.Clone();
            requiredOutputSize[dimension] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, requiredOutputSize);

            Sum(writeTarget, src, dimension);
            Div(writeTarget, writeTarget, src.Shape[dimension]);
            return writeTarget;
        }

        /// <summary>
        /// Norms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("norm", typeof(CudaStorage))]
        public NDArray Norm(NDArray result, NDArray src, int dimension, float value)
        {
            if (value == 0)
            {
                return ReductionOp.Invoke(cudaReduceKernels, "e0_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension);
            }
            else if (value == 1)
            {
                return ReductionOp.Invoke(cudaReduceKernels, "e1_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension);
            }
            else if (value == 2)
            {
                var writeTarget = ReductionOp.Invoke(cudaReduceKernels, "e2_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension);
                Pow(writeTarget, writeTarget, 0.5f);
                return writeTarget;
            }
            else
            {
                var writeTarget = ReductionOp.Invoke(cudaReduceKernels, "en_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension, value);
                Pow(writeTarget, writeTarget, 1.0f / value);
                return writeTarget;
            }
        }

        /// <summary>
        /// Standards the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("std", typeof(CudaStorage))]
        public NDArray Std(NDArray result, NDArray src, int dimension, bool normByN) { return varStdKernels.Std(result, src, dimension, normByN); }
        /// <summary>
        /// Variables the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("var", typeof(CudaStorage))]
        public NDArray Var(NDArray result, NDArray src, int dimension, bool normByN) { return varStdKernels.Var(result, src, dimension, normByN); }




        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sumall", typeof(CudaStorage))]
        public NDArray SumAll(NDArray result, NDArray src)
        {
            return ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "sumAll", result, src);
        }

        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("prodall", typeof(CudaStorage))]
        public NDArray ProdAll(NDArray result, NDArray src)
        {
            return ReduceAllOp.Invoke(cudaReduceAllKernels, 1.0f, ReduceInitType.GivenValue, "prodAll", result, src);
        }

        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("minall", typeof(CudaStorage))]
        public NDArray MinAll(NDArray result, NDArray src)
        {
            return ReduceAllOp.Invoke(cudaReduceAllKernels, 0, ReduceInitType.MaxValue, "minAll", result, src);
        }

        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("maxall", typeof(CudaStorage))]
        public NDArray MaxAll(NDArray result, NDArray src)
        {
            return ReduceAllOp.Invoke(cudaReduceAllKernels, 0, ReduceInitType.MinValue, "maxAll", result, src);
        }


        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="ArgumentException">src must be a non-empty tensor</exception>
        [RegisterOpStorageType("meanall", typeof(CudaStorage))]
        public NDArray MeanAll(NDArray result, NDArray src)
        {
            if (src.DimensionCount == 0 || src.ElementCount() == 0) throw new ArgumentException("src must be a non-empty tensor");
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            SumAll(writeTarget, src);
            Div(writeTarget, writeTarget, src.ElementCount());
            return writeTarget;
        }

        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("normall", typeof(CudaStorage))]
        public NDArray NormAll(NDArray result, NDArray src, float value)
        {
            if (value == 0)
            {
                return ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e0_norm", result, src);
            }
            else if (value == 1)
            {
                return ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e1_norm", result, src);
            }
            else if (value == 2)
            {
                var writeTarget = ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e2_norm", result, src);
                Pow(writeTarget, writeTarget, 0.5f);
                return writeTarget;
            }
            else
            {
                var writeTarget = ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "en_norm", result, src, value);
                Pow(writeTarget, writeTarget, 1.0f / value);
                return writeTarget;
            }
        }


        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="ArgumentException">src must be a non-empty tensor</exception>
        [RegisterOpStorageType("varall", typeof(CudaStorage))]
        public NDArray VarAll(NDArray result, NDArray src)
        {
            if (src.DimensionCount == 0 || src.ElementCount() == 0) throw new ArgumentException("src must be a non-empty tensor");

            var mean = Ops.MeanAll(src);
            var writeTarget = ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "en_norm", result, src, mean);
            Div(writeTarget, writeTarget, src.ElementCount() - 1);
            return writeTarget;
        }

        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("stdall", typeof(CudaStorage))]
        public NDArray StdAll(NDArray result, NDArray src)
        {
            var writeTarget = VarAll(result, src);
            Pow(writeTarget, writeTarget, 0.5f);
            return writeTarget;
        }

    }
}
