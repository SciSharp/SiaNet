// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ApplyMacros.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA.DeviceCode.Headers
{
    /// <summary>
    /// Class ApplyMacros.
    /// </summary>
    [CudaInclude("Code", "ApplyMacros")]
    public static class ApplyMacros
    {
        /// <summary>
        /// The code
        /// </summary>
        public static readonly string Code = @"

#define APPLY_T(INDEX_TYPE, DIMSA, KERNEL_NAME, OP_CODE) \
struct ConcreteOp##KERNEL_NAME { __device__ __forceinline__ void operator()(float* v) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> src, __int64 totalElements)\
    {\
        pointwiseApply1<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(src, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME());\
    }\
}

#define APPLY_TT(INDEX_TYPE, DIMSA, DIMSB, KERNEL_NAME, OP_CODE) \
struct ConcreteOp##KERNEL_NAME { __device__ __forceinline__ void operator()(float* a, float* b) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> tensorA, TensorInfo<INDEX_TYPE> tensorB, __int64 totalElements)\
    {\
        pointwiseApply2<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(tensorA, tensorB, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME());\
    }\
}

#define APPLY_TTT(INDEX_TYPE, DIMSA, DIMSB, DIMSC, KERNEL_NAME, OP_CODE) \
struct ConcreteOp##KERNEL_NAME { __device__ __forceinline__ void operator()(float* a, float* b, float *c) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> tensorA, TensorInfo<INDEX_TYPE> tensorB, TensorInfo<INDEX_TYPE> tensorC, __int64 totalElements)\
    {\
        pointwiseApply3<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB, DIMSC>(tensorA, tensorB, tensorC, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME());\
    }\
}


#define APPLY_TS(INDEX_TYPE, DIMSA, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float b;\
    __device__ ConcreteOp##KERNEL_NAME(float bVal) { this->b = bVal; }\
    __device__ __forceinline__ void operator()(float* a) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, float b, __int64 totalElements)\
    {\
        pointwiseApply1<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(a, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(b));\
    }\
}

#define APPLY_TSS(INDEX_TYPE, DIMSA, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float b;\
    float c;\
    __device__ ConcreteOp##KERNEL_NAME(float bVal, float cVal) { this->b = bVal; this->c = cVal; }\
    __device__ __forceinline__ void operator()(float* a) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, float b, float c, __int64 totalElements)\
    {\
        pointwiseApply1<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(a, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(b, c));\
    }\
}

#define APPLY_TTS(INDEX_TYPE, DIMSA, DIMSB, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float c;\
    __device__ ConcreteOp##KERNEL_NAME(float cVal) { this->c = cVal; }\
    __device__ __forceinline__ void operator()(float* a, float* b) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, TensorInfo<INDEX_TYPE> b, float c, __int64 totalElements)\
    {\
        pointwiseApply2<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(a, b, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(c));\
    }\
}

#define APPLY_TTSS(INDEX_TYPE, DIMSA, DIMSB, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float c;\
    float d;\
    __device__ ConcreteOp##KERNEL_NAME(float cVal, float dVal) { this->c = cVal; this->d = dVal; }\
    __device__ __forceinline__ void operator()(float* a, float* b) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, TensorInfo<INDEX_TYPE> b, float c, float d, __int64 totalElements)\
    {\
        pointwiseApply2<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(a, b, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(c, d));\
    }\
}


#define APPLY_TTTS(INDEX_TYPE, DIMSA, DIMSB, DIMSC, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float d;\
    __device__ ConcreteOp##KERNEL_NAME(float dVal) { this->d = dVal; }\
    __device__ __forceinline__ void operator()(float* a, float* b, float* c) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, TensorInfo<INDEX_TYPE> b, TensorInfo<INDEX_TYPE> c, float d, __int64 totalElements)\
    {\
        pointwiseApply3<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB, DIMSC>(a, b, c, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(d));\
    }\
}


";
    }
}
