// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="Math.cs" company="TensorSharp.CUDA91">
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
    /// Class MathHeader.
    /// </summary>
    [CudaInclude("Code", "Math")]
    public static class MathHeader
    {
        /// <summary>
        /// The code
        /// </summary>
        public const string Code = @"

#define INLINE_FUNC __device__ __forceinline__

//INLINE_FUNC uint8 Mod_op(uint8 x, uint8 y) { return x % y; }
INLINE_FUNC __int32 Mod_op(__int32 x, __int32 y) { return x % y; }
INLINE_FUNC float Mod_op(float x, float y) { return fmod(x, y); }
INLINE_FUNC double Mod_op(double x, double y) { return fmod(x, y); }

template<typename T> INLINE_FUNC T rsub_op(T x, T y) { return (T)(y - x); }
template<typename T> INLINE_FUNC T rdiv_op(T x, T y) { return (T)(y / x); }

#define INFIX_TO_FUNC(OPNAME, OPERATOR) template<typename T> INLINE_FUNC T OPNAME(T x, T y) { return (T)(x OPERATOR y); }
INFIX_TO_FUNC(add_op, +)
INFIX_TO_FUNC(sub_op, -)
INFIX_TO_FUNC(mul_op, *)
INFIX_TO_FUNC(div_op, /)

INFIX_TO_FUNC(gt_op, >)
INFIX_TO_FUNC(lt_op, <)
INFIX_TO_FUNC(ge_op, >=)
INFIX_TO_FUNC(le_op, <=)
INFIX_TO_FUNC(eq_op, ==)
INFIX_TO_FUNC(ne_op, !=)


template<typename T> INLINE_FUNC T Neg(T x) {
	return -x;
}

template<typename T> INLINE_FUNC T Frac(T x) {
	return x - trunc(x);
}

template<typename T> INLINE_FUNC T Lerp(T a, T b, T weight) {
	return a + weight * (b - a);
}

template<typename T> INLINE_FUNC T Sigmoid(T x) {
	return T(1) / (T(1) + exp(-x));
}


template <typename T> INLINE_FUNC T sgn(T val) {
	if (val < T(0))
		return T(-1);
	if (val > T(0))
		return T(1);
	return T(0);
}

template <typename T> INLINE_FUNC T Clamp(T val, T min, T max) {
	if (val < min)
		return min;
	if (val > max)
		return max;
	return val;
}


";
    }
}
